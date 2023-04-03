import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(abs(x1 - x2))

def KMeans(X, k):
    row_count, col_count = X.shape
    X_values = np.array(X)

    centroids = np.array(X.sample(n = k, random_state = 96))
    diff = 1

    Xd = np.empty([k,row_count])

    while(diff!=0):

        i=0
        for index1, row_c in enumerate(centroids):
            ED = []
            for index2, row_d in enumerate(X_values):
                d = euclidean_distance(row_c, row_d)
                ED.append(d)
            Xd[i] = ED
            i +=1
            
        X["Cluster"] = np.argmin(Xd, axis=0)+1
        centroids_new = X.groupby(["Cluster"]).mean()
        
        diff = (np.subtract(centroids_new, centroids)).sum()
        diff = diff.sum()

        centroids = np.array(centroids_new)

    return X, centroids

def KMedians(X, k):
    row_count, col_count = X.shape
    X_values = np.array(X)

    centroids = np.array(X.sample(n = k, random_state = 63))
    diff = 1

    Xd = np.empty([k,row_count])

    while(diff!=0):

        i=0
        for index1, row_c in enumerate(centroids):
            MD = []
            for index2, row_d in enumerate(X_values):
                d = manhattan_distance(row_c, row_d)
                MD.append(d)
            Xd[i] = MD
            i +=1
            
        X["Cluster"] = np.argmin(Xd, axis=0)+1
        
        centroids_new = X.groupby(["Cluster"]).median()
        
        diff = (np.subtract(centroids_new, centroids)).sum()
        diff = diff.sum()

        centroids = np.array(centroids_new)

    return X, centroids       

def graph_plot(k, p, r, f):
    plt.plot(k, p, label='Precision')
    plt.plot(k, r, label='Recall')
    plt.plot(k, f, label='F-score')  
    plt.xlabel('K')
    plt.legend()  
    plt.show()



if __name__ == "__main__":
    animals = pd.read_csv('data/animals', delimiter=' ', header=None)
    countries = pd.read_csv('data/countries', delimiter=' ', header=None)
    fruits = pd.read_csv('data/fruits', delimiter=' ', header=None)
    veggies = pd.read_csv('data/veggies', delimiter=' ', header=None)

    animals[0] = np.ones(len(animals[0]))
    countries[0] = np.ones(len(countries[0]))+1
    fruits[0] = np.ones(len(fruits[0]))+2
    veggies[0] = np.ones(len(veggies[0]))+3

    df = pd.concat([animals,countries,fruits,veggies],ignore_index=True)
    # print(df.shape)

    label = df[0].astype(int)
    label_count = df.groupby(0).size()
    label = np.array(label)

    ########
    #          K-Means
    ########

    print('K-Means')

    precision_all =  []
    recall_all = []
    Fscore_all = []
    for i in range(1,10):
        data = df.drop(df.columns[0], axis=1)
        new_data, cent = KMeans(data, k=i)

        new_label = np.array(new_data['Cluster'].astype(int))
        new_label_count = new_data.groupby(["Cluster"]).size().astype(int)

        precision =  np.empty([len(new_label)])
        recall = np.empty([len(new_label)])
        Fscore = np.empty([len(new_label)])

        for x,y in enumerate(new_label):
            count = 0
            for p,q in enumerate(new_label):
                if(q==y and label[x]==label[p]):
                    count += 1
            precision[x] = count/new_label_count[y]
            recall[x] = count/label_count[label[x]]
            Fscore[x] = (2*precision[x]*recall[x])/(precision[x]+recall[x])
        precision_all.append(precision.mean())
        recall_all.append(recall.mean())
        Fscore_all.append(Fscore.mean())
        print(i, precision.mean(), recall.mean(), Fscore.mean())
    
    k = range(1,10)  
    plt.title('K-Means')
    graph_plot(k, precision_all, recall_all, Fscore_all)

    ########
    #          K-Means with Normalization
    ########

    print('K-Means with Normalization')

    precision_all =  []
    recall_all = []
    Fscore_all = []
    for i in range(1,10):
        d = df.drop(df.columns[0], axis=1)
        
        data = d / np.sqrt(np.sum(d**2))
                   
        new_data, cent = KMeans(data, k=i)

        new_label = np.array(new_data['Cluster'].astype(int))
        new_label_count = new_data.groupby(["Cluster"]).size().astype(int)

        precision =  np.empty([len(new_label)])
        recall = np.empty([len(new_label)])
        Fscore = np.empty([len(new_label)])

        for x,y in enumerate(new_label):
            count = 0
            for p,q in enumerate(new_label):
                if(q==y and label[x]==label[p]):
                    count += 1
            precision[x] = count/new_label_count[y]
            recall[x] = count/label_count[label[x]]
            Fscore[x] = (2*precision[x]*recall[x])/(precision[x]+recall[x])
        precision_all.append(precision.mean())
        recall_all.append(recall.mean())
        Fscore_all.append(Fscore.mean())
        print(i, precision.mean(), recall.mean(), Fscore.mean())
    
    k = range(1,10)  
    plt.title('K-Means with Normalization')
    graph_plot(k, precision_all, recall_all, Fscore_all)



    ########
    #          K-Medians
    ########

    print('K-Medians')

    precision_all =  []
    recall_all = []
    Fscore_all = []
    for i in range(1,10):
        data = df.drop(df.columns[0], axis=1)
        new_data, cent = KMedians(data, k=i)

        new_label = np.array(new_data['Cluster'].astype(int))
        new_label_count = new_data.groupby(["Cluster"]).size().astype(int)

        precision =  np.empty([len(new_label)])
        recall = np.empty([len(new_label)])
        Fscore = np.empty([len(new_label)])

        for x,y in enumerate(new_label):
            count = 0
            for p,q in enumerate(new_label):
                if(q==y and label[x]==label[p]):
                    count += 1
            precision[x] = count/new_label_count[y]
            recall[x] = count/label_count[label[x]]
            Fscore[x] = (2*precision[x]*recall[x])/(precision[x]+recall[x])
        precision_all.append(precision.mean())
        recall_all.append(recall.mean())
        Fscore_all.append(Fscore.mean())
        print(i, precision.mean(), recall.mean(), Fscore.mean())
    
    k = range(1,10)  
    plt.title('K-Medians')
    graph_plot(k, precision_all, recall_all, Fscore_all)

    ########
    #          K-Medians with Normalization
    ########

    print('K-Medians with Normalization')

    precision_all =  []
    recall_all = []
    Fscore_all = []
    for i in range(1,10):
        d = df.drop(df.columns[0], axis=1)        
        data = d / np.sqrt(np.sum(d**2))
        new_data, cent = KMedians(data, k=i)

        new_label = np.array(new_data['Cluster'].astype(int))
        new_label_count = new_data.groupby(["Cluster"]).size().astype(int)

        precision =  np.empty([len(new_label)])
        recall = np.empty([len(new_label)])
        Fscore = np.empty([len(new_label)])

        for x,y in enumerate(new_label):
            count = 0
            for p,q in enumerate(new_label):
                if(q==y and label[x]==label[p]):
                    count += 1
            precision[x] = count/new_label_count[y]
            recall[x] = count/label_count[label[x]]
            Fscore[x] = (2*precision[x]*recall[x])/(precision[x]+recall[x])
        precision_all.append(precision.mean())
        recall_all.append(recall.mean())
        Fscore_all.append(Fscore.mean())
        print(i, precision.mean(), recall.mean(), Fscore.mean())
    
    k = range(1,10)  
    plt.title('K-Medians with Normalization')
    graph_plot(k, precision_all, recall_all, Fscore_all)
