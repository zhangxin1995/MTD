import logging
from module.eval_metrics import evaluate,rankscore,evaluate_args,evaluate_reranking,build_evaluate,evaluate_market_args,eval_PTR_map,tc,tds2
import torch
import numpy as np
import torch.nn as nn
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pickle as pkl
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist,pdist
import math
import torch.nn.functional as F
from tqdm import tqdm
import time
from scipy.io import savemat
def process_ts(ts,eta=1):
    #PTD       1544889600000
    # new_ts=(ts)/100/86400*math.pi
    #Market
    # print(eta)
    # exit()
    new_ts=(ts-1544889600000)/100/86400*eta #*math.pi
    return new_ts
def norm(t):
    if t.size(0)!=1:
        raise('error')
    return F.normalize(t,p=2)
class TrackletEmbedding(nn.Module):
    def sinusoidal_position_embedding(self,t, output_dim, device):
        '''编码函数
        Args:
            t (int): 要编码的位置
            output_dim (int): 特征维度
            device : GPU编号
        Returns: 
            embeddings : sin cos编码信息
        '''
        position =  t 
        embeddings = position * self.w  
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.to(device)
        return embeddings
    
    def RoPE(self,q,t,c,output_dim=None):
       
        if output_dim is None:
            output_dim = q.shape[-1]
        # print(q.size())
        q_norm=q/torch.norm(q.unsqueeze(0), p=2, dim=1, keepdim=True).squeeze()
        # return q_norm
        # print(q_norm.size())
        # exit()
        pos_emb = self.sinusoidal_position_embedding(t, output_dim, q.device)
        cos_pos = pos_emb[:,1].repeat_interleave(2, dim=-1) 
        sin_pos = pos_emb[:,0].repeat_interleave(2, dim=-1) 
        q2 = torch.stack([-q_norm[..., 1::2], q_norm[..., ::2]], dim=-1)
        q2 = q2.reshape(q.shape) 
        f = q_norm * cos_pos + q2 * sin_pos  #+ self.embedding(torch.tensor(c).cuda())
        f/=torch.norm(f.unsqueeze(0), p=2, dim=1, keepdim=True).squeeze()
        return f.unsqueeze(0)
    
    def __init__(self,dim=1024,seed=3):
        super(TrackletEmbedding,self).__init__()
        self.w=torch.Tensor(int(dim/2))
        self.w=nn.Parameter(self.w,requires_grad=True)
        # self.embedding=nn.embedding(11)
        self.setup_seed(seed)
        nn.init.normal(self.w)
        # nn.init.constant_(self.w,1)
        self.embedding=nn.Embedding(11,1024)

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # random.seed(seed)    
    def forward(self,features,ts,cs):
        fetures=features.cuda()
        st_embedding_features=torch.zeros_like(features)
        for i in range(features.size()[0]):
            st_embedding_features[i,:]=self.RoPE(features[i,:],ts[i],cs[i])
        return st_embedding_features

class VehicleRouteResolution:
    def __init__(self):
        self.manage=pywrapcp.RoutingIndexManager
    def print_solution(self,dist, num, manager, routing, solution):
        """Prints solution on console."""
        print(f'Objective: {solution.ObjectiveValue()}')
        total_distance = 0
        for vehicle_id in range(num):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                plan_output += ' {} ->'.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            plan_output += ' {}\n'.format(manager.IndexToNode(index))
            plan_output += 'Distance of the route: {}\n'.format(route_distance)
            print(plan_output)
            total_distance += route_distance
        print('Total Distance of all routes: {}'.format(total_distance))
        
    def retrieval_solution(self,dist, num, manager, routing, solution):
        """Prints solution on console."""
        # print(dist)
        total_distance = 0
        paths=[]
        for vehicle_id in range(num):
            # print('num',num)
            index = routing.Start(vehicle_id)
            route_distance = 0
            path=[]
            while not routing.IsEnd(index):
                # plan_output += ' {} ->'.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                if manager.IndexToNode(index)!=0:
                    path.append(manager.IndexToNode(index)-1)
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            total_distance += route_distance
            paths.append(path) 
        return paths,total_distance
    
    def vr(self,dist,num,end_weights=None):

        manager = pywrapcp.RoutingIndexManager(dist.shape[0],num, 0)
        routing = pywrapcp.RoutingModel(manager)
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if end_weights is not None and to_node==0 and from_node!=0:
                return end_weights[from_node-1]
            return dist[from_node][to_node]
        # print(dist)
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        dimension_name = 'Distance'
        routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        int(2e10),  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(1000)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = 30
        # search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
        search_parameters.first_solution_strategy = (routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        solution = routing.SolveWithParameters(search_parameters)
        if solution is None:
            return False,[[i] for i in range(dist.shape[0]-1)],1e5
        a,b=self.retrieval_solution(dist, num, manager, routing, solution)
        return True,a,b


            
    def trajectory_generation(self,dist,trajectory_num,weight=-1,weights=None,para=1):
        sub_paths=[]
        ds=[]
        min_d=1e10
        min_d_idx=0
        l=1
        r=dist.shape[0]+1
        # print(dist)
        for i in range(1,r):
            su,sub_path,d=self.sub_trajectory_generation(dist,i,weight=weight,weights=weights)                
            sub_paths.append(sub_path)
            if not su:
                return sub_path,d
            ds.append(d)
            if d<min_d:
                min_d_idx=i-1
                min_d=d
        # for i in range(len(ds)):
        #     print(i+1,ds[i],sub_paths[i])
        # input()
        # print(ds,min_d_idx)
        return sub_paths[min_d_idx],ds[min_d_idx]
    
    def sub_trajectory_generation(self,dist,trajectory_num,weight=-1,weights=None,scale=1e2):
        if weights is not None:
            start_weights=weights[0]
            end_weight=weights[1]*scale
            new_dist=np.ones((dist.shape[0]+1,dist.shape[1]+1))*0.5/2
            for i in range(1,new_dist.shape[0]):
                new_dist[0,i]=start_weights[i-1]
                new_dist[i,0]=start_weights[i-1]
        elif weight!=-1:
            new_dist=np.ones((dist.shape[0]+1,dist.shape[1]+1))*weight/2
        else:
            new_dist=np.ones((dist.shape[0]+1,dist.shape[1]+1))*0.5/2
        new_dist[1:,1:]=dist.data.cpu()
        new_dist[0,0]=0
        
        # print(new_dist) 
        # input() 
        # print('-----------------')
        # print(new_dist)
        # print('-----------------')
        if trajectory_num==0:
            # for num in range(1,new_dist.shape[0]):
            su,paths,d=self.vr(new_dist*scale,new_dist.shape[0]-1,end_weights=end_weights)
            return su,paths,d
        else:
            su,paths,d=self.vr(new_dist*scale,trajectory_num)
            return su,paths,d
        
                        # None,real_dist,self.cfg,ts,trajectory_num=self.cfg.DATALOADER.TRAIN_SIZE
    def __call__(self, features, dist, weight=-1, weights=None,trajectory_num=0,para=0):
        block_paths_idx,d=self.trajectory_generation(dist,trajectory_num,weight=weight,weights=weights,para=para)
        if features is not None:
            block_paths_idx,block_paths_features=self.retrieval_features(block_paths_idx,features)
        else:
            block_paths_features=None
        return block_paths_idx, block_paths_features

    def retrieval_features(self,paths,features):
        #print('features',features.size(),len(paths))
        idx_from_features=[]
        feat=torch.cat([f.unsqueeze(0) for f in features],0)
        new_paths=[]
        for p in paths:
            if len(p)!=0:
                new_paths.append(p)
        paths=new_paths
        for path in paths:
            if len(path)==1:
                idx_from_features.append(feat[path[0],:].unsqueeze(0)/torch.norm(feat[path[0],:],dim=0))
            else:
                temp=[]
                for j in path:
                    temp.append(feat[j,:].unsqueeze(0))
                f=torch.mean(torch.cat(temp,dim=0),dim=0).unsqueeze(0)
                f=f/torch.norm(f)
                idx_from_features.append(f)
        temp=[]
        # print(len(idx_from_features))
        # exit()
        # idx_from_features=torch.cat(idx_from_features,dim=0)
        return paths,idx_from_features
                



def fcluster_adj2path_from_dist(dist,threshold=0.2,criterion='distance'):
    # print(dist)
    if len(dist.shape)==2:
        new_dist=[]
        for i in range(dist.shape[0]):
            for j in range(i+1,dist.shape[0]):
                new_dist.append(dist[i,j])
        new_dist=np.array(new_dist)
    else:
        new_dist=dist
    link = linkage(new_dist, "average")
    clusters = fcluster(link,threshold, criterion=criterion)
    paths=defaultdict(list)
    for m,c in enumerate(clusters):
        paths[c-1].append(m)
    return paths
         
            
            
def extract_feat_ts_dist(model,feat,ts,cs,blocking=True):
    max_value=1e6
    encode_feats=model(feat,ts,cs)
    encode_feats = encode_feats  #/ torch.norm(encode_feats, dim=-1, keepdim=True)
    dist=build_evaluate(encode_feats, encode_feats, 'cosine',need_norm=False) 
    if blocking:
        for i in range(len(ts)):
            for j in range(len(ts)):
                if ts[i]>ts[j]:
                    dist[i,j]=max_value
    return encode_feats,dist           
        
def manifold_ptd_main(config):
    epoch=0
    trajectory_extract_model=VehicleRouteResolution()
    
    start_cam_weights=np.zeros(11)
    end_cam_weights=np.zeros(11)
    cam2idx={'SQ0923':0, 
         'SQ0930':1,
         'SQ0927':2,
         'SQ0931':3,
         'SQ0922':4,
         'SQ0925':5,
         'SQ0921':6,
         'SQ0926':7,
         'SQ0924':8
        } 
   
    with open(config['dataset']['path'],'rb') as infile:
        datas=pkl.load(infile,encoding='latin')
    qfs=torch.Tensor(datas['qfs'])
    embedding_model=TrackletEmbedding(dim=qfs[0,:].size(0),seed=config['seed'])
    qpids=torch.Tensor(datas['qls'])
    qcs=torch.Tensor([cam2idx[c] for c in datas['qcs']])
    qts=torch.Tensor(datas['qts'])
    gfs=torch.Tensor(datas['tfs'])
    gtidxs=torch.Tensor(datas['tidxs'])
    gpids=torch.Tensor(datas['tls'])
    gts=torch.Tensor(datas['gts'])
    gcs=torch.Tensor([cam2idx[c] for c in datas['tcs']])
    ranks=[1,5,10]
    cmc, mAP = evaluate_reranking(qfs, qpids, qcs, gfs, gpids, gcs, ranks, 'cosine')
    print("[EPOCH {}]Standard Video Retrieval Test Results ----------{}".format(epoch,0))
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    # exit()
    # g_t_dist=build_evaluate(gfs, train_fs, 'cosine').cpu().numpy() 
    # q_g_dist=build_evaluate(qfs, gfs, 'cosine').cpu().numpy() 
    # q_t_dist=build_evaluate(qfs, train_fs, 'cosine').cpu().numpy() 
    g_g_dist=build_evaluate(gfs, gfs, 'cosine').cpu().numpy()  
    # t_t_dist=build_evaluate(train_fs, train_fs, 'cosine').cpu().numpy()          
    time_start=time.time()
    threshold=config['cluster_threshold'] #0.046
    criterion='distance'
    candidate_set=fcluster_adj2path_from_dist(g_g_dist,threshold=threshold,criterion=criterion)
    candidate_set_path=[]
    candidate_set_f=[]
    # print('len',len(candidate_set))
    # exit()
    
    values_correct=[]
    values_t_correct=[]
    values_error=[]
    values_t_error=[]
    for key in candidate_set.keys():
        path=candidate_set[key]
        # print(path)
        # print(len(path))
        if len(path)==1:
            # values_correct.append(0)
            pass
        else:
            for i in range(len(path)):
                for j in range(i+1,len(path)):
                    if gpids[path[i]]==gpids[path[j]]:
                        values_correct.append(g_g_dist[path[i],path[j]])
                        values_t_correct.append((gts[path[i]]-gts[path[j]]).item())
                    else:
                        values_error.append(g_g_dist[path[i],path[j]])
                        values_t_error.append((gts[path[i]]-gts[path[j]]).item())
        # test_fms,test_fes=draw_hist([values_correct,values_error],tag='test_feature hist {}'.format(epoch))
        # test_tms,test_tes=draw_hist([values_t_correct,values_t_error],tag='test_time hist {}'.format(epoch))
    values_correct=[]
    values_t_correct=[]
    values_error=[]
    values_t_error=[]
    
    for key in tqdm(candidate_set.keys()):
        path=candidate_set[key]
        features=gfs[path,:]
        # print(path)
        ts=process_ts(gts[path],eta=config['manifold']['eta'])
        cs=[int(gcs[i]) for i in path]
        pa=config['manifold']['alpha']
        sw=np.array([pa for c in cs])
        ew=np.array([pa for c in cs])
        # sw=np.array([(1-start_cam_weights[c])*pa for c in cs])
        # ew=np.array([(1-end_cam_weights[c])*pa for c in cs])
        ws=[sw,ew]
        # print(ws)
        # input()
        # if len(path)==1:
        #     subf=norm(gfs[path[0],:].unsqueeze(0))
        # else:
        #     f=torch.cat([gfs[i,:].unsqueeze(0) for i in path],dim=0)
        #     f=torch.mean(f,dim=0).unsqueeze(0)
        #     subf=norm(f)
        # candidate_set_f.append(subf)
        # candidate_set_path.append(path)  
        # continue
        encode_feats,real_dist= extract_feat_ts_dist(embedding_model,features,ts,cs)
        # print(real_dist)
        # input()
        paths,sub_features = trajectory_extract_model(None,real_dist,weights=ws)
        # print(paths)
        sub_pids=[]
        for p in paths:
            if len(p)==1:
                subf=norm(gfs[path[p[0]],:].unsqueeze(0))
            elif len(p)==0:
                continue
            else:
                f=torch.cat([gfs[path[i],:].unsqueeze(0) for i in p],dim=0)
                f=torch.mean(f,dim=0).unsqueeze(0)
                subf=norm(f)
                for i in range(len(p)):
                    for j in range(i+1,len(p)):
                        if gpids[path[p[i]]]==gpids[path[p[j]]]:
                            values_correct.append(real_dist[p[i],p[j]].item())
                            values_t_correct.append((gts[path[p[i]]]-gts[path[p[j]]]).item())
                        else:
                            values_error.append(real_dist[p[i],p[j]].item())
                            values_t_error.append((gts[path[p[i]]]-gts[path[p[j]]]).item())
            candidate_set_f.append(subf)
            candidate_set_path.append([path[i] for i in p])  
            sub_pids.append([int(gpids[path[i]]) for i in p])
            # print(sub_pids[-1])
        # print(sub_pids[-1])
        # input()        
    # print('values',len(values_correct),len(values_error))
    time_end=time.time()
    print('time',time_end-time_start)
    candidate_set_f=torch.cat(candidate_set_f,dim=0)    
    paths=candidate_set_path
    features=candidate_set_f.cpu().numpy()
    adist=cdist(qfs.cpu().numpy(),features,metric='cosine')
    indices = np.argsort(adist, axis=1)
    indices=indices.astype(np.int64)
    args1=[]
    args2=[]
    dist3=[]
    for i in tqdm(range(adist.shape[0])):
        args=np.argsort(adist[i,:])
        args3=[]
        temp=[]
        aidxs=[]
        for j in range(adist.shape[1]):
            path=paths[args[j]]
            subf  = torch.cat([gfs[k,:].unsqueeze(0) for k in path],dim=0)
            dist2 = cdist(qfs[i,:].unsqueeze(0).cpu(),subf.cpu())
            args4 = np.argsort(dist2[0,:])
            for k in args4:
                args3.append(path[k])
            temp.extend(dist2[0,:].tolist())
            aidxs.extend([path[j] for j in args4])
        idxs2=[]
        temp2=[]
        for j,item in enumerate(aidxs):
            if item not in idxs2:
                idxs2.append(item)
                temp2.append(temp[j])
        args2.append(args3)
        args1.append(args)
        dist3.append(temp2)
    cmc2, all_AP2, all_INP2 = evaluate_market_args(args2, qpids.cpu().numpy(),gpids.cpu().numpy(), qcs.cpu().numpy(), gcs.cpu().numpy(), 20)
    mAP2 = np.mean(all_AP2)
    mINP2 = np.mean(all_INP2)
    print("[EPOCH: {}]Trajectory Test Results ----------{}".format(i,0))
    print("mAP: {:.1%}".format(mAP2))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc2[r - 1]))
    print("------------------")
    # for candidate_path  in candidate_set_path:
    #     sub_gts=gts[candidate_path]
    idx2pathidx={}
    tpath2index=[]
    for i,idx in enumerate(gpids):
        pid=int(idx)
        if pid in idx2pathidx.keys():
            tpath2index[idx2pathidx[pid][0]].append(i)
        else:
            idx2pathidx[pid]=[len(tpath2index)]
            tpath2index.append([i])     
    with open('tr_distmat.pkl','wb') as out:
        pkl.dump({'dist2':dist3,'args':args2,'idx_from_gfs':paths,'dist':adist,'args2':indices,'idx2pathidx':idx2pathidx,'tpath2index':tpath2index},out)                   
    gls=[]
    for path in paths:
        gls.append([int(gpids[j]) for j in path])
    m=eval_PTR_map(indices,qpids,gls)
    print("PTR mAP: {:.1%}".format(m))
    gls=[]
    for path in paths:
        gls.append([int(gpids[j]) for j in path])
    m=eval_PTR_map(indices,qpids,gls) 
    print("PTR mAP: {:.1%}".format(m))
    n=tc(qpids,gpids,paths)
    print("TLC: {:.1%}".format(n))
    z=tds2(qpids,idx2pathidx,tpath2index,paths,topk=10)
    print("TDS: {:.1%}".format(z))
    tas=m*n*z
    print("TAS: {:.1%}".format(tas))
    with open('tr_distmat.pkl','wb') as out:
        pkl.dump({'dist2':dist3,'args':args2,'idx_from_gfs':paths,'dist':adist,'args2':indices,'idx2pathidx':idx2pathidx,'tpath2index':tpath2index},out)                   
    exit()
    
