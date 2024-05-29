from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import torch
import numpy as np
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
        int(1e10),  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(1000)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        # search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
        search_parameters.first_solution_strategy = (routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        solution = routing.SolveWithParameters(search_parameters)
        # print(type(solution))
        # self.print_solution(dist, num, manager, routing, solution)
        # return 
        # print(solution)
        return self.retrieval_solution(dist, num, manager, routing, solution)

            
    def trajectory_generation(self,dist,trajectory_num,weight=-1,weights=None,para=1):
        # print(dist)
        # input()
        # sub_path,d=self.sub_trajectory_generation(dist, trajectory_num,weight=weight)
        sub_paths=[]
        ds=[]
        min_d=1e10
        min_d_idx=0
        l=1
        r=dist.shape[0]+1
        # print(dist)
        for i in range(1,r):
            sub_path,d=self.sub_trajectory_generation(dist,i,weight=weight,weights=weights)
            sub_paths.append(sub_path)
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
        max_dist=1e6
        if trajectory_num==0:
            # for num in range(1,new_dist.shape[0]):
            paths,d=self.vr(new_dist*scale,new_dist.shape[0]-1,end_weights=end_weights)
                # if d<max_dist:
                #     max_dist=d
                #     max_paths=paths
                # else:
                #     break        
                # # print('vr',paths,num,d)
            return paths,d
        else:
            paths,d=self.vr(new_dist*scale,trajectory_num)
            return paths,d
        
                        # None,real_dist,self.cfg,ts,trajectory_num=self.cfg.DATALOADER.TRAIN_SIZE
    def __call__(self, features, dist, cfg,weight=-1, weights=None,trajectory_num=0,para=0):
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
                


