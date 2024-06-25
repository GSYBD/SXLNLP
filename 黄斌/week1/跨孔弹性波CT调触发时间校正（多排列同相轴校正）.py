import numpy as np
import math
# 将CT剖面时间文件导入并转换为数组
file='C:/Users/Administrator/Desktop/MVNZ4-GB-518_MVNZ4-GB-515.txt'
array=np.loadtxt(file)
# 计算CT剖面接收点排列数
geophone_depth_max=max(array[:,5])
geophone_depth_min=min(array[:,5])
print(geophone_depth_max)
print(geophone_depth_min)
node_number=math.ceil((geophone_depth_max-geophone_depth_min+1)/24)
print(node_number)
# 计算接收点排列节点前后的检波点深度
def node(max,number):
    node_dep=[]
    current_number=1
    while current_number<number:
        node_front_depth=geophone_depth_max-24*current_number+1
        node_back_depth=geophone_depth_max-24*current_number
        node_dep.append([node_front_depth,node_back_depth])
        current_number+=current_number
    return(node_dep)
node_depth=[]
node_depth=node(geophone_depth_max,node_number)
print(node_depth)
# 计算炮点个数及接收点道数
shot_point=max(array[:,1])
number_of_track=max(array[:,0])
print(shot_point)
print(number_of_track)
#统计各炮记录接收点节点处同相轴递增或递减,节点处错断类型（上浮或下沉），错断节点前的平均时差，错断节点前后时差等
def trend(point,track,number,depth,arr):
    sign_1=np.arange(1,point+1,1)
    print(sign_1)
    current_number=1
    while current_number<number:
        sign_2=np.zeros(int(point))
        sign_3=np.full(int(point),2)
        sign_4=np.zeros(int(point))
        sign_5=np.zeros(int(point))
        dep=depth[current_number-1]
        a=dep[0]
        b=dep[1]
        for i in arr[:,0]:
            j=int(i-1)
            if int(arr[j,5])==int(a):
                d1=abs(arr[j,5]-arr[j,3])
                d2=abs(arr[j+1,5]-arr[j+1,3])
                t1=arr[j,6]
                t2=arr[j+1,6]
                sign_4[int(arr[j,1]-1)]=abs(t2-t1)
                dt=(max(arr[j-5:j,6])-min(arr[j-5:j,6]))/len(arr[j-5:j,6])
                sign_5[int(arr[j,1]-1)]=dt
                print(dt)
                if d1<d2:
                    sign_2[int(arr[j,1]-1)]=1
                if abs(t2-t1)>2*dt:
                    if t2>t1:
                        sign_3[int(arr[j,1]-1)]=3
                    else:
                        sign_3[int(arr[j,1]-1)]=4            
        print(a)
        sign1=np.vstack((sign_1,sign_2,sign_3,sign_4,sign_5))
        current_number+=current_number
        return(sign1)
sign=trend(shot_point,number_of_track,node_number,node_depth,array)
print(sign)                         
# 确定CT记录各炮集同相轴错断节点后各道的触发时间校正值
def tri(sig):
    sig_row=np.size(sig,0)
    sig_col=np.size(sig,1)
    d=int((sig_row-1)/4)
    l=int(sig_col)
    print(l)
    sig_1=sig[0,:]
    sig_2=np.zeros(l)
    while d>=1:
        for i in sig[0,:]:
            j=int(i-1)
            print(j)
            if int(sig[4*d-2,j])==4:
                   if int(sig[4*d-3,j])==0:
                       sig_2[j]=-sig[4*d-1,j]+sig[4*d,j]
                   if int(sig[4*d-3,j])==1:
                       sig_2[j]=sig[4*d-1,j]+sig[4*d,j]
            if int(sig[4*d-2,j])==3:
                   if int(sig[4*d-3,j])==0:
                       sig_2[j]=-(sig[4*d-1,j]+sig[4*d,j])
                   if int(sig[4*d-3,j])==1:
                       sig_2[j]=-sig[4*d-1,j]+sig[4*d,j]
                   sig1=np.vstack((sig_1,sig_2))
        d-=d
    return(sig1)
triggering=tri(sign)           
print(triggering)
# 对CT记录中有错断同相轴的炮集记录进行时间校正
def correction(trig,track,number,depth,arr):
    trig_row=np.size(trig,0)
    trig_col=np.size(trig,1)
    d=int(trig_row)
    l=int(trig_col)
    while number>1:
        dep=depth[number-2]
        a=dep[0]
        for i in trig[0,:]:
            k=int(i-1)
            for j in arr[:,0]:
                m=int(j-1)
                if int(arr[m,1])==int(trig[0,k]):
                    if arr[m,5]<a:
                        arr[m,6]=arr[m,6]+trig[d+1-number,k]
        number-=number
    return arr
correct_time=correction(triggering,number_of_track,node_number,node_depth,array)
np.savetxt('C:/Users/Administrator/Desktop/CT.txt',correct_time,fmt="%.3f",delimiter='  ')
        
                   
        
        
    

                 
             
             

