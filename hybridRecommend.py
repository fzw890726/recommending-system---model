 #! /usr/bin/python
"""
*This script implements a Recommender system which combines user-based collaborative
filtering and populairty approach.
*Run the script under anaconda is suggested.
*Author: Xiaoling Chen
*Date:12/23/16

"""
import pandas as pd
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import scipy.sparse
import csv


class RecommendSys():
    def __init__(self, input1,input2):
        '''initialization'''
        x = self.process_input_file(input1,input2)
        whole_df,group_df,itemIDs,userIDs,group_cate_df = x
        self.whole_df=whole_df   
        self.group_df = group_df 
        self.itemIDs = itemIDs
        self.userIDs = userIDs
        self.group_cate_df = group_cate_df
        self.bu = None   #average rating for each user
        self.utility_matrix = self.construct_utility_matrix()
        
    def process_input_file(self,input1,input2):
        df=pd.read_csv(input1)
        df2=pd.read_csv(input2)
        frames = [df,df2]
        whole_df = pd.concat(frames)
        #add a column for each transcation and count the number of transcation for each customer and each item
        whole_df['rates']=whole_df['Customer Number'].map(lambda x: 1 )
        group_df=whole_df.groupby(by=['Customer Number','Item Number'])['rates'].sum()
        itemIDs= whole_df['Item Number'].unique()
        userIDs = whole_df['Customer Number'].unique()
        group_cate_df=whole_df.groupby(by=['NPD Sysco Segmentation','Item Number'])['rates'].sum()
        return whole_df,group_df,itemIDs,userIDs,group_cate_df
    
    def get_popular_item_for_category(self,category,K):
        if isinstance(category,float): #handle case category=Nan
            return []
        d =  self.group_cate_df[category].to_dict()
        a = sorted(d.items(), key=lambda x: x[1],reverse=True)[:K]
        predicted_item = [i[0] for i in a]
        return predicted_item

    def category_rating(self,number):
        """normalize the transaction times to rating from 1-10"""
        group = [0,5,10,15,20,25,30,35,40,50] #10 groups
        rating = 0
        for i in range(len(group)):
            if number >=group[len(group)-1-i]:
                rating = len(group)-i
                return rating
        return rating
    
    def construct_utility_matrix(self):
        """construct utility matrix using sparse matrix and calcalate the average rating for each user"""
        row = []
        col = []
        data = []
        mshape = (len(self.userIDs), len(self.itemIDs))
        for row_index in range(len(self.userIDs)):
            userID = self.userIDs[row_index]
            items_for_userID = self.group_df[userID].keys()
            for item in items_for_userID:
                column_index = self.itemIDs.tolist().index(item)
                rating = self.category_rating(self.group_df[userID][item])
                row.append(row_index)
                col.append(column_index)
                data.append(rating)
        utility_sparse_matrix=csr_matrix((np.asarray(data),(np.asarray(row),np.asarray(col))),
                                       shape=mshape) 
        u = sum(data)/float(len(data))
        (x,y,z)=scipy.sparse.find(utility_sparse_matrix)
        countings=np.bincount(x)
        sums=np.bincount(x,weights=z)
        bu=sums/countings-u
        self.bu = bu
        utility_matrix = utility_sparse_matrix.todense()
        return utility_matrix
    
    def get_similarity_matrix(self):
        """generate similarity matrix using Pearson"""
        dist_out = np.corrcoef(self.utility_matrix)
        return dist_out
    
    def get_cosine_similarity_matrix(self):
        """generate similarity matrix using cosine"""
        from sklearn.metrics import pairwise_distances
        from scipy.spatial.distance import cosine
        dist_out = 1-pairwise_distances(self.utility_matrix, metric="cosine")
        return dist_out
    
    def user_base_CF(self,user_index,N,dist_out):
        """predict for each user using user-base CF"""
        a=sorted(range(len(dist_out[user_index])), key=lambda j: dist_out[user_index][j])[-N-1:]
        topN_other_user_index = a[0:N][::-1]
        ru = self.bu[user_index]
        Rxi = np.zeros((1,len(self.itemIDs)))
        summary = 0
        for other_user_index in topN_other_user_index:
            Rxi = Rxi+dist_out[user_index][other_user_index]*(self.utility_matrix[other_user_index,:]-self.bu[other_user_index])
            summary+=abs(dist_out[user_index][other_user_index])
        Rxi = ru+Rxi/summary
        Rxi = Rxi.tolist()[0]
        #get rid of the items that already purpase in 2014 and 2015
        newRxi = []
        for j in range(len(Rxi)):
            if self.utility_matrix[user_index,j]>0:
                newRxi.append(0)
            else:
                newRxi.append(Rxi[j])
        predict_item_index= sorted(range(len(newRxi)), key=lambda j: newRxi[j])[-10:][::-1]
        predict_item_name = [self.itemIDs[i] for i in predict_item_index]
        return predict_item_name
        
    def get_predict_for_user(self,userID,category,N,dist_out):
        """predict 10 new items for one userID"""
        x=None
        if userID in self.userIDs:  #exsiting user
            user_index = self.userIDs.tolist().index(userID)
            x = self.user_base_CF(user_index,N,dist_out)
        else:  #new user
            x = self.get_popular_item_for_category(category,10)
        return x
    
    def process_test_file(self,testfile):
        test_df=pd.read_csv(testfile)
        test_df['rates']=test_df['Customer Number'].map(lambda x: 1 )
        group_test_df=test_df.groupby(by=['Customer Number','Item Number'])['rates'].sum()  
        return test_df,group_test_df

    def test_precision_hybrid(self,N,dist_out,testfile):
        """test the hybrid model precision @10 on testfile"""
        test_df,group_test_df = self.process_test_file(testfile)
        test_userIDs = test_df['Customer Number'].unique()
        sum_precision = 0
        c=0
        for userID in test_userIDs:
            category = test_df[test_df['Customer Number']==userID]['NPD Sysco Segmentation'].unique()[0]
            x = self.get_predict_for_user(userID,category,N,dist_out)
            items_for_userID = group_test_df[userID].keys()
            number_of_purpase = 10-len(set(x)-set(items_for_userID))
            precision = number_of_purpase/10.0
            c=c+1
            sum_precision+=precision
        print 'number of user',c
        print 'average precision',sum_precision/c
        
    def run_prediction_for_all_users(self,N,dist_out,outfile,testfile):
        test_df=pd.read_csv(testfile)
        test_userIDs = test_df['Customer Number'].unique()
        all_userIDs = list(set(self.userIDs.tolist()+test_userIDs.tolist()))
        f=file(outfile,'wt')
        writer = csv.writer(f)
        writer.writerow( ('Customer ID','Recommend items') )
        for userID in all_userIDs:
            if userID in test_userIDs:
                category = test_df[test_df['Customer Number']==userID]['NPD Sysco Segmentation'].unique()[0]
            else:
                category = self.whole_df[self.whole_df['Customer Number']==userID]['NPD Sysco Segmentation'].unique()[0]
            x = self.get_predict_for_user(userID,category,N,dist_out)
            x = tuple([str(userID)]+[str(i) for i in x])
            writer.writerow( x )
        f.close()
        print 'write to outfile done!'
        
def main():
   """
- To run the script successfully, input files for training the model needs provided in the main function.
- N is the number K-NN nearest neighborhood and can be changed.
- dist_out is the similarity matrix and can be changed between cosine or Pearson.
- The prediction results are written to a csv file
"""
    input1 = r"/Users/xchen2/Documents/sysco_folder/Sysco Case Study/Sales Transactions - 2014.csv"
    input2 = r"/Users/xchen2/Documents/sysco_folder/Sysco Case Study/Sales Transactions - 2015.csv"
    testfile = r"/Users/xchen2/Documents/sysco_folder/Sysco Case Study/Sales Transactions - 2016.csv"
    Recommender = RecommendSys(input1,input2)
    dist_out = Recommender.get_similarity_matrix()  #Pearson
    #dist_out = Recommender.get_cosine_similarity_matrix()    #cosine
    N=85      #number of neighbors in the user-based CF model
    Recommender.test_precision_hybrid(N,dist_out,testfile)  #evaluate model on test file
    #generate recommend item list for all the user in 2014,2015 and 2016 and write to csv file
    Recommender.run_prediction_for_all_users(N,dist_out,'prediction_for_all_users.csv',testfile)
       
   
if __name__ == '__main__': 
   main()  
