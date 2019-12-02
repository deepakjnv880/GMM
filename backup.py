from math import log as ln
import numpy as np
from matplotlib import pyplot as plt
import math
import random
dimention=3
percent_training_data=0.75
K=3
def compute_mean_vector(data_vectors,mean_vector,gamma):
    mean_vector=np.zeros((K,2))
    for k in range(K):
        # mean_vector=[0,0]
        sum=0

        for i in range(len(data_vectors)):
            # print(gamma[i][k],"=== ",data_vectors[i])
            mean_vector[k][0]+=(gamma[i][k]*(float(data_vectors[i][0])))
            mean_vector[k][1]+=(gamma[i][k]*(float(data_vectors[i][1])))
            sum+=gamma[i][k]
        mean_vector[k][0]/=(sum)
        mean_vector[k][1]/=(sum)
    return mean_vector

def compute_covariance_matrix(data_vectors,mean_vector,gamma):
    # print(gamma)
    covariance_matrix=np.zeros((K,2,2))
    for k in range(K):
        for i in range(dimention):
            for j in range(dimention):
                sum=0
                for x in range(len(data_vectors)):
                    covariance_matrix[k][i][j]+=((gamma[x][k])*(data_vectors[x][i]-mean_vector[k][i])*(data_vectors[x][j]-mean_vector[k][j]))
                    sum+=gamma[x][k]
                covariance_matrix[k][i][j]/=(sum);
    return covariance_matrix

def computer_pi(data_vectors,gamma):
    pi=[0 for v in range(K)]
    for k in range(K):
        for i in range(len(data_vectors)):
            pi[k]+=gamma[i][k]
        pi[k]/=len(data_vectors)
    return pi


def read_training_dataset(filename):
    file_object=open(filename,'r')
    file_data=file_object.readlines()
    data_vectors=[]#list of list
    for line in file_data:
        data_vectors.append([float(x) for x in line.split()])
    return data_vectors

def evaluate_gaussian(x,cov,mean):
    # print(cov,np.linalg.det(cov)," === ",cov)
    x=np.array(x)
    mean=np.array(mean)
    # temp=[x[0]-mean[0],x[1]-mean[1]]
    cov=np.array(cov)
    k=math.exp((-1/2)*(np.matmul((np.subtract(x,mean)),np.matmul(np.linalg.inv(cov),np.transpose(np.subtract(x,mean))))))
    k/=math.sqrt(2*3.14*np.linalg.det(cov));
    return k

def computer_gamma(data_vectors,mean_vector,covariance_matrix,pi_vector):
    gamma_vector=np.zeros((len(data_vectors),K))
    for i in range(len(data_vectors)):
        sum=0
        for k in range(K):
            gamma_vector[i][k]=evaluate_gaussian(data_vectors[i],covariance_matrix[k],mean_vector[k])
            sum+=gamma_vector[i][k]
        for k in range(K):
            gamma_vector[i][k]/=sum
    return gamma_vector

def compute_loglikelihood(data_vectors,mean_vector,covariance_matrix,pi_vector):
    total=0
    for i in range(len(data_vectors)):
        temp=0
        for k in range(K):
            temp=pi_vector[k]*evaluate_gaussian(data_vectors[i],covariance_matrix[k],mean_vector[k])
        temp=ln(temp)
        total+=temp
    return total


def main():
     mean_vector=[]
     covariance_matrix=[[0 for x in range(dimention)] for y in range(dimention)]
     training_data=[[] for i in range(dimention)]
     # testing_data_X=[[] for i in range(3)]
     # testing_data_Y=[[] for i in range(3)]
     # xmin=ymin=100;xmax=ymax=-100
     upper_limit=-100
     lower_limit=100
     data_vectors=[]
     for i in range(1,4):
         filename='data_nls/Class'+str(i)+'.txt'
         data_vectors=read_training_dataset(filename)
         # print 'mini ==============================='
         # print
         lower_limit=min(lower_limit, np.amin(data_vectors));
         upper_limit=max(upper_limit, np.amax(data_vectors));
         for item in data_vectors[:int(len(data_vectors))]:
             # print item
             training_data[i-1].append([item[0],item[1],i-1])
             # training_data_Y[i-1].append(item[1])
         # for item in data_vectors[int(len(data_vectors)*percent_training_data):]:
         #     # print item
         #     testing_data.append([item[0],item[1],i-1])
     # print(len(data_vectors))
     # mean_vector=np.zeros(K)
     mean_vector=[]
     for v in range(K):
         mean_vector.append(random.choice(data_vectors))
         pass
     # covariance_matrix=np.zeros((K,2,2))
     covariance_matrix=[[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]]
     # pi_vector=np.zeros(K)
     pi_vector=[0,0,0]
     gamma=np.zeros((len(data_vectors),K))
     # for x in range(100):
     #     print("iteration number = ",x)
     #     gamma=computer_gamma(data_vectors,mean_vector,covariance_matrix,pi_vector)
     #     mean_vector=compute_mean_vector(data_vectors,mean_vector,gamma)
     #     covariance_matrix=compute_covariance_matrix(data_vectors,mean_vector,gamma)
     #     pi_vector=computer_pi(data_vectors,gamma)
     #     log_likelihood=compute_loglikelihood(data_vectors,mean_vector,covariance_matrix,pi_vector)
     #     print("log_likelihood == ",log_likelihood)

     # print(gamma)
     colour=["red","blue","green"]
     print ('### Drawing graph ###\n')
     # print X,Y
     #naming the x axis
     plt.xlabel('x - axis')
     # naming the y axis
     plt.ylabel('y - axis')
     plt.scatter((np.array(training_data[0]).T)[0],(np.array(training_data[0]).T)[1],s=1 ,color='red',label='class 1')
     plt.scatter((np.array(training_data[1]).T)[0],(np.array(training_data[1]).T)[1],s=1, color='green',label='class 2')
     plt.scatter((np.array(training_data[2]).T)[0],(np.array(training_data[2]).T)[1],s=1, color='blue',label='class 3')

     # for i in range(len(data_vectors)):
     #     output=((gamma[i]).tolist()).index(max(gamma[i]))
     #     plt.plot(data_vectors[i][0], data_vectors[i][1],marker='o', markersize=1, color=colour[output])

     # plt.legend()
     plt.savefig('result.png')
     # plt.savefig('result.pdf')
     plt.show()
     # print("Model accuracy is ",float((len(testing_data)-wrongly_classfied)/len(testing_data))*100,"%\n")




if __name__ == '__main__':
    main()
    # print("frist")
