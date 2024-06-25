import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
import torch.nn.functional as F
from torch_geometric.nn import to_hetero, GATConv, Linear
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
seed = 42
np.random.seed(seed=seed)
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

def load_edge(x, y, data):
    nums = len(data)
    index = np.ones((2, nums))
    for i in range(nums):
        index[0][i] = np.where(x == data[i][0])[0][0]
        index[1][i] = np.where(y == data[i][1])[0][0]
    edge_index = torch.tensor(index, dtype=torch.long)
    edge_label = torch.ones((nums, 1))
    return edge_index, edge_label

def load_weight_edge(matrix):
    index = np.nonzero(matrix)
    nums = np.count_nonzero(matrix)
    edge_index = torch.tensor(index, dtype=torch.long)
    edge_label = torch.zeros(nums,1)
    for i in range(nums):
        edge_label[i][0] = torch.tensor(matrix[index[0][i]][index[1][i]])
    return edge_index, edge_label

def avg_ass(ass):
    pre = 0
    rec = 0
    f1 = 0
    n = len(ass)
    for i in range(n):
        pre += ass[i]['precision']
        rec += ass[i]['recall']
        f1 += ass[i]['f1-score']
    return pre/n,rec/n,f1/n

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels_1, out_channels_1, heads):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels_1*heads, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels_1*heads)
        self.conv2 = GATConv((-1, -1), out_channels_1*heads, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels_1*heads)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = F.normalize(x, dim=1)
        return x

miRNA = pd.read_csv('./data_processed/miRNA_names.csv', header=0)['miRNA name'].values
disease = pd.read_csv('./data_processed/disease_names.csv', header=0)['disease name'].values
drug = pd.read_csv('./data_processed/drug_names.csv', header=0)['drug name'].values
miRNA_disease = pd.read_csv('./data_processed/miRNA_disease.csv', header=0).iloc[:, :].values
miRNA_drug = pd.read_csv('./data_processed/miRNA_drug.csv', header=0).iloc[:, :].values
drug_disease = pd.read_csv('./data_processed/drug_disease.csv', header=0).iloc[:, :].values
miRNA_sim_miRNA = pd.read_csv('./data_processed/miRNA_similarity_matrix.csv', header=0).values
disease_sim_disease = pd.read_csv('./data_processed/disease_similarity_matrix.csv', header=0).values
drug_sim_drug = pd.read_csv('./data_processed/drug_similarity_matrix.csv', header=0).values
miRNA_gs_miRNA = pd.read_csv('./data_processed/miRNA_gs_matrix.csv', header=0).values
disease_gs_disease = pd.read_csv('./data_processed/disease_gs_matrix.csv', header=0).values
drug_gs_drug = pd.read_csv('./data_processed/drug_gs_matrix.csv', header=0).values

miRNA_disease_edge_index, miRNA_disease_edge_attr = load_edge(miRNA, disease, miRNA_disease)
miRNA_drug_edge_index, miRNA_drug_edge_attr = load_edge(miRNA, drug, miRNA_drug)
drug_disease_edge_index, drug_disease_edge_attr = load_edge(drug, disease, drug_disease)
miRNA_sim_miRNA_edge_index, miRNA_sim_miRNA_edge_attr = load_weight_edge(miRNA_sim_miRNA)
disease_sim_disease_edge_index, disease_sim_disease_edge_attr = load_weight_edge(disease_sim_disease)
drug_sim_drug_edge_index, drug_sim_drug_edge_attr = load_weight_edge(drug_sim_drug)
miRNA_gs_miRNA_edge_index, miRNA_gs_miRNA_edge_attr = load_weight_edge(miRNA_gs_miRNA)
disease_gs_disease_edge_index, disease_gs_disease_edge_attr = load_weight_edge(disease_gs_disease)
drug_gs_drug_edge_index, drug_gs_drug_edge_attr = load_weight_edge(drug_gs_drug)

d_list = [64, 128, 256]
hc_oc_list = [(64, 8), (64, 16), (64, 32), (64, 64), 
         (128, 16), (128, 32), (128, 64), (128, 128),
         (256, 32), (256, 64), (256, 128), (256, 256)]
heads = 1

for d in d_list:
    for hc_oc in hc_oc_list:

        start = time.time()
        file = open(f"./result/HBNMM.txt", mode='a+')              
        file.write(time.strftime('%Y-%m-%d %H:%M:%S \n', time.localtime(time.time())))
        file.write(f'd:{d} hidden_channels:{hc_oc[0]} out_channels:{hc_oc[1]} heads:{heads}\n')
        file.write('GAT2  sim+gs \n')
        # plt.figure()

        # init data
        drug_node_feature = torch.rand((len(drug), d), dtype=torch.float)
        miRNA_node_feature = torch.rand((len(miRNA), d), dtype=torch.float)
        disease_node_feature = torch.rand((len(disease), d), dtype=torch.float)
        drug_node_feature = F.normalize(drug_node_feature, dim=1)
        miRNA_node_feature = F.normalize(miRNA_node_feature, dim=1)
        disease_node_feature = F.normalize(disease_node_feature, dim=1)

        data = HeteroData()
        data['miRNA'].x = miRNA_node_feature
        data['drug'].x = drug_node_feature
        data['disease'].x = disease_node_feature
        data['miRNA', 'association', 'disease'].edge_index = miRNA_disease_edge_index
        data['miRNA', 'association', 'drug'].edge_index = miRNA_drug_edge_index
        data['drug', 'association', 'disease'].edge_index = drug_disease_edge_index
        data['miRNA', 'similar', 'miRNA'].edge_index = miRNA_sim_miRNA_edge_index
        data['disease', 'similar', 'disease'].edge_index = disease_sim_disease_edge_index
        data['drug', 'similar', 'drug'].edge_index = drug_sim_drug_edge_index
        data['miRNA', 'gauss', 'miRNA'].edge_index = miRNA_gs_miRNA_edge_index
        data['disease', 'gauss', 'disease'].edge_index = disease_gs_disease_edge_index
        data['drug', 'gauss', 'drug'].edge_index = drug_gs_drug_edge_index
        data['miRNA', 'association', 'disease'].edge_attr = miRNA_disease_edge_attr
        data['miRNA', 'association', 'drug'].edge_attr = miRNA_drug_edge_attr
        data['drug', 'association', 'disease'].edge_attr = drug_disease_edge_attr
        data['miRNA', 'similar', 'miRNA'].edge_attr = miRNA_sim_miRNA_edge_attr
        data['disease', 'similar', 'disease'].edge_attr = disease_sim_disease_edge_attr
        data['drug', 'similar', 'drug'].edge_attr = drug_sim_drug_edge_attr
        data['miRNA', 'gauss', 'miRNA'].edge_attr = miRNA_gs_miRNA_edge_attr
        data['disease', 'gauss', 'disease'].edge_attr = disease_gs_disease_edge_attr
        data['drug', 'gauss', 'drug'].edge_attr = drug_gs_drug_edge_attr
        data = T.ToUndirected(merge=False)(data)

        model = GNN(hidden_channels_1=hc_oc[0], out_channels_1=hc_oc[1], heads=heads)
        model = to_hetero(model, data.metadata(), aggr='sum')

        loss_avg = 0
        miRNA_drug_auc = []
        miRNA_disease_auc = []
        drug_disease_auc = []
        all_auc = []
        miRNA_drug_ass = []
        miRNA_disease_ass = []
        drug_disease_ass = []
        all_ass = []
        miRNA_drug_acc = []
        miRNA_disease_acc = []
        drug_disease_acc = []
        all_acc = []
        threshold = 0
        fpr_list = []
        tpr_list = []

        for fold in range(5):
            data = data.cuda()
            # negative sample
            transform = RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=True, edge_types=[('miRNA', 'association', 'drug'), ('miRNA', 'association', 'disease'), ('drug', 'association', 'disease')], 
                                    rev_edge_types=[('drug', 'rev_association', 'miRNA'), ('disease', 'rev_association', 'miRNA'), ('disease', 'rev_association', 'drug')])
            train_data, val_data, test_data = transform(data)
            train_data = train_data.cuda()
            test_data = test_data.cuda()
            model = model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            loss_list = []
            auc_list = []
            loss_min = 100000000

            train_label = torch.concat([train_data['miRNA', 'association', 'drug'].edge_label,train_data['miRNA', 'association', 'disease'].edge_label,
                                        train_data['drug', 'association', 'disease'].edge_label], dim=0)
            train_miRNA_drug_size = len(train_data['miRNA', 'association', 'drug'].edge_label)
            train_miRNA_disease_size = len(train_data['miRNA', 'association', 'disease'].edge_label)
            train_drug_disease_size = len(train_data['drug', 'association', 'disease'].edge_label)

            model.train()
            for epoch in range(1000):
                optimizer.zero_grad()
                out = model(data.x_dict, train_data.edge_index_dict)
                preds_size = train_miRNA_drug_size + train_miRNA_disease_size + train_drug_disease_size
                preds = torch.zeros(preds_size).cuda()

                # train_miRNA_drug_preds = torch.zeros(train_miRNA_drug_size).cuda()
                for i in range(train_miRNA_drug_size):
                    m = train_data['miRNA', 'association', 'drug'].edge_label_index[0][i]
                    n = train_data['miRNA', 'association', 'drug'].edge_label_index[1][i]
                    pred = torch.dot(out['miRNA'][m], out['drug'][n])
                    # train_miRNA_drug_preds[i] = pred
                    preds[i] = pred
                
                # train_miRNA_disease_preds = torch.zeros(train_miRNA_disease_size).cuda()
                for i in range(train_miRNA_disease_size):
                    m = train_data['miRNA', 'association', 'disease'].edge_label_index[0][i]
                    n = train_data['miRNA', 'association', 'disease'].edge_label_index[1][i]
                    pred = torch.dot(out['miRNA'][m], out['disease'][n])
                    # train_miRNA_disease_preds[i] = pred
                    preds[i+train_miRNA_drug_size] = pred

                # train_drug_disease_preds = torch.zeros(train_drug_disease_size).cuda()
                for i in range(train_drug_disease_size):
                    m = train_data['drug', 'association', 'disease'].edge_label_index[0][i]
                    n = train_data['drug', 'association', 'disease'].edge_label_index[1][i]
                    pred = torch.dot(out['drug'][m], out['disease'][n])
                    # train_drug_disease_preds[i] = pred
                    preds[i+train_miRNA_drug_size+train_miRNA_disease_size] = pred

                loss = F.binary_cross_entropy_with_logits(preds, train_label)
                # if epoch % 100 ==0:
                #     print(f"The {epoch} epoch loss is: {loss}")
                # loss_list.append(loss.cpu().detach().numpy())
                if loss < loss_min:
                    loss_min = loss
                    torch.save({'model': model.state_dict()}, f'./result/{d}_{hc_oc[0]}_{hc_oc[1]}_{fold}_h{heads}_HBNMM.pth')
                
                loss.backward()
                optimizer.step()

            loss_avg += loss_min

            test_label = torch.concat([test_data['miRNA', 'association', 'drug'].edge_label,test_data['miRNA', 'association', 'disease'].edge_label,
                                        test_data['drug', 'association', 'disease'].edge_label], dim=0)
            test_miRNA_drug_size = len(test_data['miRNA', 'association', 'drug'].edge_label)
            test_miRNA_disease_size = len(test_data['miRNA', 'association', 'disease'].edge_label)
            test_drug_disease_size = len(test_data['drug', 'association', 'disease'].edge_label)
            
            model.eval()
            with torch.no_grad():
                state_dict = torch.load(f'./result/{d}_{hc_oc[0]}_{hc_oc[1]}_{fold}_h{heads}_HBNMM.pth')
                model.load_state_dict(state_dict['model'])
                test_out = model(data.x_dict, test_data.edge_index_dict)

                test_miRNA_drug_preds = torch.zeros(test_miRNA_drug_size)
                test_miRNA_drug_01 = torch.zeros(test_miRNA_drug_size)
                for i in range(test_miRNA_drug_size):
                    m = test_data['miRNA', 'association', 'drug'].edge_label_index[0][i]
                    n = test_data['miRNA', 'association', 'drug'].edge_label_index[1][i]
                    test_miRNA_drug_preds[i] = torch.dot(test_out['miRNA'][m], test_out['drug'][n])
                auc = roc_auc_score(test_data['miRNA', 'association', 'drug'].edge_label.cpu(), test_miRNA_drug_preds)
                file.write(f"The {fold}th fold's miRNA-drug auc is:{auc}\n")
                miRNA_drug_auc.append(auc)
                fpr, tpr, thresholds = roc_curve(test_data['miRNA', 'association', 'drug'].edge_label.cpu(), test_miRNA_drug_preds)
                for i in range(len(fpr)):
                    if abs(tpr[i]+fpr[i]-1) <= 0.005:
                        threshold = thresholds[i]
                        break
                for i in range(test_miRNA_drug_size):
                    m = test_data['miRNA', 'association', 'drug'].edge_label_index[0][i]
                    n = test_data['miRNA', 'association', 'drug'].edge_label_index[1][i]
                    if torch.dot(test_out['miRNA'][m], test_out['drug'][n]) <= threshold:
                        test_miRNA_drug_01[i] = 0
                    else:
                        test_miRNA_drug_01[i] = 1
                ass = classification_report(test_data['miRNA', 'association', 'drug'].edge_label.cpu(), test_miRNA_drug_01, output_dict=True)
                file.write(f"The {fold}th fold's miRNA-drug pre is:{ass['1.0']['precision']}\n")
                file.write(f"The {fold}th fold's miRNA-drug rec is:{ass['1.0']['recall']}\n")
                file.write(f"The {fold}th fold's miRNA-drug f1 is:{ass['1.0']['f1-score']}\n")
                file.write(f"The {fold}th fold's miRNA-drug acc is:{ass['accuracy']}\n")
                miRNA_drug_ass.append(ass['1.0'])
                miRNA_drug_acc.append(ass['accuracy'])
                
                test_miRNA_disease_preds = torch.zeros(test_miRNA_disease_size)
                test_miRNA_disease_01 = torch.zeros(test_miRNA_disease_size)
                for i in range(test_miRNA_disease_size):
                    m = test_data['miRNA', 'association', 'disease'].edge_label_index[0][i]
                    n = test_data['miRNA', 'association', 'disease'].edge_label_index[1][i]
                    test_miRNA_disease_preds[i] = torch.dot(test_out['miRNA'][m], test_out['disease'][n])
                auc = roc_auc_score(test_data['miRNA', 'association', 'disease'].edge_label.cpu(), test_miRNA_disease_preds)
                file.write(f"The {fold}th fold's miRNA-disease auc is:{auc}\n")
                miRNA_disease_auc.append(auc)
                fpr, tpr, thresholds = roc_curve(test_data['miRNA', 'association', 'disease'].edge_label.cpu(), test_miRNA_disease_preds)
                for i in range(len(fpr)):
                    if abs(tpr[i]+fpr[i]-1) <= 0.005:
                        threshold = thresholds[i]
                        break
                for i in range(test_miRNA_disease_size):
                    m = test_data['miRNA', 'association', 'disease'].edge_label_index[0][i]
                    n = test_data['miRNA', 'association', 'disease'].edge_label_index[1][i]
                    test_miRNA_disease_preds[i] = torch.dot(test_out['miRNA'][m], test_out['disease'][n])
                    if torch.dot(test_out['miRNA'][m], test_out['disease'][n]) <= threshold:
                        test_miRNA_disease_01[i] = 0
                    else:
                        test_miRNA_disease_01[i] = 1
                ass = classification_report(test_data['miRNA', 'association', 'disease'].edge_label.cpu(), test_miRNA_disease_01, output_dict=True)
                file.write(f"The {fold}th fold's miRNA-disease pre is:{ass['1.0']['precision']}\n")
                file.write(f"The {fold}th fold's miRNA-disease rec is:{ass['1.0']['recall']}\n")
                file.write(f"The {fold}th fold's miRNA-disease f1 is:{ass['1.0']['f1-score']}\n")
                file.write(f"The {fold}th fold's miRNA-disease acc is:{ass['accuracy']}\n")
                miRNA_disease_ass.append(ass['1.0'])
                miRNA_disease_acc.append(ass['accuracy'])

                test_drug_disease_preds = torch.zeros(test_drug_disease_size)
                test_drug_disease_01 = torch.zeros(test_drug_disease_size)
                for i in range(test_drug_disease_size):
                    m = test_data['drug', 'association', 'disease'].edge_label_index[0][i]
                    n = test_data['drug', 'association', 'disease'].edge_label_index[1][i]
                    test_drug_disease_preds[i] = torch.dot(test_out['drug'][m], test_out['disease'][n])
                auc = roc_auc_score(test_data['drug', 'association', 'disease'].edge_label.cpu(), test_drug_disease_preds)
                file.write(f"The {fold}th fold's drug-disease auc is:{auc}\n")
                drug_disease_auc.append(auc)
                fpr, tpr, thresholds = roc_curve(test_data['drug', 'association', 'disease'].edge_label.cpu(), test_drug_disease_preds)
                for i in range(len(fpr)):
                    if abs(tpr[i]+fpr[i]-1) <= 0.005:
                        threshold = thresholds[i]
                        break
                for i in range(test_drug_disease_size):
                    m = test_data['drug', 'association', 'disease'].edge_label_index[0][i]
                    n = test_data['drug', 'association', 'disease'].edge_label_index[1][i]
                    if torch.dot(test_out['drug'][m], test_out['disease'][n]) <= threshold:
                        test_drug_disease_01[i] = 0
                    else:
                        test_drug_disease_01[i] = 1
                ass = classification_report(test_data['drug', 'association', 'disease'].edge_label.cpu(), test_drug_disease_01, output_dict=True)
                file.write(f"The {fold}th fold's drug-disease pre is:{ass['1.0']['precision']}\n")
                file.write(f"The {fold}th fold's drug-disease rec is:{ass['1.0']['recall']}\n")
                file.write(f"The {fold}th fold's drug-disease f1 is:{ass['1.0']['f1-score']}\n")
                file.write(f"The {fold}th fold's drug-disease acc is:{ass['accuracy']}\n")
                drug_disease_ass.append(ass['1.0'])
                drug_disease_acc.append(ass['accuracy'])
                
                test_preds = torch.concat([test_miRNA_drug_preds, test_miRNA_disease_preds, test_drug_disease_preds], dim=0)
                test_01 = torch.concat([test_miRNA_drug_01, test_miRNA_disease_01, test_drug_disease_01], dim=0)
                fpr, tpr, thresholds = roc_curve(test_label.cpu(), test_preds)
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                auc = roc_auc_score(test_label.cpu(), test_preds)
                file.write(f"The {fold}th fold's all auc is:{auc}\n")
                all_auc.append(auc)
                ass = classification_report(test_label.cpu(), test_01, output_dict=True)
                file.write(f"The {fold}th fold's all pre is:{ass['1.0']['precision']}\n")
                file.write(f"The {fold}th fold's all rec is:{ass['1.0']['recall']}\n")
                file.write(f"The {fold}th fold's all f1 is:{ass['1.0']['f1-score']}\n")
                file.write(f"The {fold}th fold's all acc is:{ass['accuracy']}\n")
                all_ass.append(ass['1.0'])
                all_acc.append(ass['accuracy'])

                print(f"The {fold}th fold's min loss is: {loss_min}")
                print(f"The {fold}th fold's best auc: {auc}") 
                file.write(f"The {fold}th fold's min loss is: {loss_min}\n")
                file.write(f"The {fold}th fold's best auc: {auc}\n")
                # x = np.arange(1000)
                # y = loss_list
               
                # plt.plot(x, y)
                # plt.ylabel("loss")
                # plt.xlabel("epoch")
                # plt.savefig(f'./figs/loss_{d}_{hc_oc[0]}_{hc_oc[1]}_{fold}_h{heads}_GAT3.tiff', dpi=600)

        end = time.time()  
        min = int((end-start)/60)
        sec = end-start-min*60     
        file.write(f"average loss is {loss_avg/5}\n")
        file.write("-----miRNA-drug--------\n")
        file.write(f"auc:{sum(miRNA_drug_auc)/len(miRNA_drug_auc)},")
        pre, rec, f1 = avg_ass(miRNA_drug_ass)
        file.write(f"pre:{pre},rec:{rec},f1:{f1},acc:{sum(miRNA_drug_acc)/len(miRNA_drug_acc)}\n")

        file.write("-----miRNA-disease-----\n")
        file.write(f"auc:{sum(miRNA_disease_auc)/len(miRNA_disease_auc)},")
        pre, rec, f1 = avg_ass(miRNA_disease_ass)
        file.write(f"pre:{pre},rec:{rec},f1:{f1},acc:{sum(miRNA_disease_acc)/len(miRNA_disease_acc)}\n")
    
        file.write("-----drug-disease------\n")
        file.write(f"auc:{sum(drug_disease_auc)/len(drug_disease_auc)},")
        pre, rec, f1 = avg_ass(drug_disease_ass)
        file.write(f"pre:{pre},rec:{rec},f1:{f1},acc:{sum(drug_disease_acc)/len(drug_disease_acc)}\n")
        
        file.write("-------all-------------\n")
        file.write(f"auc:{sum(all_auc)/len(all_auc)},")
        pre, rec, f1 = avg_ass(all_ass)
        file.write(f"pre:{pre},rec:{rec},f1:{f1},acc:{sum(all_acc)/len(all_acc)}\n")
    
        file.write(f"total running time is: {min}min {sec}sec \n \n")
        file.close()

        plt.figure()
        plt.title("ROC curves of HBNMM in 5-fold validation")
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot(fpr_list[0],tpr_list[0],label="fold 1")
        plt.plot(fpr_list[1],tpr_list[1],label="fold 2")
        plt.plot(fpr_list[2],tpr_list[2],label="fold 3")
        plt.plot(fpr_list[3],tpr_list[3],label="fold 4")
        plt.plot(fpr_list[4],tpr_list[4],label="fold 5")
        avg_fpr = [(x1+x2+x3+x4+x5)/5 for x1,x2,x3,x4,x5 in zip(fpr_list[0],fpr_list[1],fpr_list[2],fpr_list[3],fpr_list[4])]
        avg_tpr = [(y1+y2+y3+y4+y5)/5 for y1,y2,y3,y4,y5 in zip(tpr_list[0],tpr_list[1],tpr_list[2],tpr_list[3],tpr_list[4])]
        plt.plot(avg_fpr,avg_tpr,label="Mean")
        plt.legend(loc='lower right')
        plt.savefig(f'./figs/AUC.tiff', dpi=600)
        plt.show()