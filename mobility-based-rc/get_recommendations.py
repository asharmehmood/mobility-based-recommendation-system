import pandas as pd
from surprise import SVD
import numpy as np
import surprise
from surprise import Reader, Dataset
from ast import literal_eval

total_recom=5

def referesh_database():
	from scipy import spatial
	r_data=pd.read_csv('/home/ashar/fiverr/mobility-based-rc/full-data.csv',usecols=['user_number','content number','title','genre','loc','Weekdays/Weekends','Gender','Profession','Age','age','event','view','time','device','section','CP','watch time ( candle )','t-gap'])

	dont_insert=['new_old']
	inp_file=pd.read_csv('./input_file_new_user.csv')
	columns=inp_file.columns
	for index,row in inp_file.iterrows():
	    r_data.loc[len(r_data)] = [row['user_id'],row['content_id'],'none',row['genre'],row['loc'],row['weekdays/weekends'],row['gender'],row['profession'],row['age'],row['age_category'],row['event'],row['view'],row['time'],row['device'],row['section'],row['cp'],row['watch_time'],0]

	r_data.to_csv('./full-data2.csv')
	r_data=pd.read_csv('/home/ashar/fiverr/mobility-based-rc/full-data2.csv',usecols=['user_number','content number','title','genre','loc','Weekdays/Weekends','Gender','Profession','Age','age','event','view','time','device','section','CP','watch time ( candle )','t-gap'])

	#replace content number with title (numerical)
	id_to_title=pd.DataFrame(columns=['id','title'])
	unique_title=list(r_data['title'].unique())
	title_id={}
	count=1
	for t in unique_title:
	    title_id[t.strip()]=count
	    id_to_title.loc[len(id_to_title)]=[count,t.strip()]
	    count+=1

	id_to_title.to_csv('./id_to_title.csv')
	for t in title_id:
	    r_data.loc[r_data['title'].str.strip()==t,'title']=title_id[t]

	r_data=r_data.drop('content number', 1)
	r_data=r_data.rename(columns={'title': 'content number'})

	#genre selected after preprocessing of dataset
	genre=['documentary','quiz','contest','audition','history','news','sports','drama','romantic relationship',
	      'reality','sitcom','action','talk','family','comedy','sf fantasy','sf','mid','adventure','romance','trip',
	      'game','fantasy','mini series','horror','crime','mystery']
	for g in genre:
	    r_data[g]=0
	    
	for index,row in r_data.iterrows():
	    cur_g=row['genre'].split(',')
	    for g in cur_g:
	        if g.strip().lower() in genre:
                    r_data.loc[index,g.strip().lower()] = 1
		    
	r_data=r_data.drop(['genre'], axis = 1)
	r_data.rename(columns = {'age':'age_category','content number':'content_id','user_number':'user_id','watch time ( candle )':'watch_time','t-gap':'t_gap'},inplace=True)
	r_data.columns= r_data.columns.str.lower()
	r_data.to_csv('./ready_to_train_data.csv')  #ready to train for matrix factorization

	#rating scale calculation
	min_rate=r_data['watch_time'].min()
	max_rate=r_data['watch_time'].max()

	###matrix factorization (collaborative filtering)
	print("Matrix factorization started...")
	# It is to specify how to read the data frame.
	reader = Reader(rating_scale=(min_rate,max_rate+1))
	train_data_mf = Dataset.load_from_df(r_data[['user_id', 'content_id', 'watch_time']], reader)
	#It is of dataset format from surprise library
	trainset = train_data_mf.build_full_trainset()
	svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
	svd.fit(trainset)
	#predictions of SVD which will be used later as a feature
	train_preds = svd.test(trainset.build_testset())
	train_pred_mf = np.array([pred.est for pred in train_preds])
	print("Matrix factorization successfully ended.")
	#### Hand crafted features (related to watch time)
	r_data=pd.read_csv('./ready_to_train_data.csv')
	#1 average watch time of all series by all users
	g_avg=r_data['watch_time'].mean()
	r_data['g_avg']=g_avg
	#2 average watch time of a movie by all users
	r_data['m_avg']=r_data.groupby(['content_id'])['watch_time'].transform('mean')
	#3 average watch time of all movies watched by a user
	r_data['u_avg']=r_data.groupby(['user_id'])['watch_time'].transform('mean')
	#4 average watch time of a movie by a user
	r_data['um_avg']=r_data.groupby(['user_id','content_id'])['watch_time'].transform('mean')

	### Hand crafted features (similar user and content, get similar user watch time and add as feature)
	user_content_features=['gender','profession','age','age_category','section','cp','documentary','quiz','contest','audition','history','news','sports','drama','romantic relationship',
	      'reality','sitcom','action','talk','family','comedy','sf fantasy','sf','mid','adventure','romance','trip',
	      'game','fantasy','mini series','horror','crime','mystery']

	unique_user=list(r_data['user_id'].unique())

	#get feature average of every user
	user_feature_avg={}
	for u in unique_user:
	    user_feature_avg[u]=[]
	    for f in user_content_features:
                user_feature_avg[u].append(r_data.loc[r_data['user_id']==u][f].mean())

	#get cosine similarity between every user and sort them
	u_cosim={}
	for u in user_feature_avg:
	    u_cosim[u]={}
	    for u2 in user_feature_avg:
                if u!=u2:
                    sim = 1 - spatial.distance.cosine(user_feature_avg[u], user_feature_avg[u2])
                    u_cosim[u][u2]=sim
	    u_cosim[u]=dict( sorted(u_cosim[u].items(),key=lambda item: item[1],reverse=True))  

	#get top five user for every user
	top_5={}
	top=5
	for u in u_cosim:
	    top_5[u]=[]
	    top_cur=0
	    for su in u_cosim[u]:
                top_5[u].append(su)
                top_cur+=1
                if top_cur==top:
                    break

	#add top five user average watch time of a content as another feature
	print("Encircling neighbour users...")
	r_data['simu1']=1
	r_data['simu2']=2
	r_data['simu3']=3
	r_data['simu4']=4
	r_data['simu5']=5
	for index,row in r_data.iterrows():
	    if index%500==0:
                print(index)
	    user=int(row['user_id'])
	    content=int(row['content_id'])
	    sim_u=top_5[user]
	    for ind,u in enumerate(sim_u):
                wtime=r_data.loc[(r_data['user_id']==u) & (r_data['content_id']==content)]['um_avg']
                simu_wtime=0
                if len(wtime)==0:
                    simu_wtime=0
                else:
                    simu_wtime=wtime.iloc[0]
                if ind==0:
                    r_data.at[index,'simu1']=simu_wtime
                elif ind==1:
                    r_data.at[index,'simu2']=simu_wtime
                elif ind==2:
                    r_data.at[index,'simu3']=simu_wtime
                elif ind==3:
                    r_data.at[index,'simu4']=simu_wtime
                elif ind==4:
                    r_data.at[index,'simu5']=simu_wtime
	print("Found some users")
	#add matrix factorization prediction as final feature
	r_data['mat_fact_pred']=train_pred_mf

	#data ready for recommendation extraction
	r_data.to_csv('./final_data.csv')

	r_data=pd.read_csv('./final_data.csv')
	localities=[1,2,3,4,5,6,7]

	## make production file
	prod_field=['loc','weekdays/weekends','gender','profession','age','age_category','event','view','time','device',
	    'section','cp','watch_time','t_gap','documentary','quiz','contest','audition','history','news','sports','drama',
		'romantic relationship','reality','sitcom','action','talk','family','comedy','sf fantasy',
		'sf','mid','adventure','romance','trip','game','fantasy','mini series','horror','crime','mystery',
		'g_avg','m_avg','u_avg','um_avg','simu1','simu2','simu3','simu4','simu5','mat_fact_pred']

	unique_user=list(r_data['user_id'].unique())

	#get feature average of every user
	user_feature_avg={}
	for u in unique_user:
	    user_feature_avg[u]=[]
	    for f in prod_field:
                user_feature_avg[u].append(r_data.loc[r_data['user_id']==u][f].mean())
		
	#get cosine similarity between every user and sort them
	from scipy import spatial

	u_cosim={}
	for u in user_feature_avg:
	    u_cosim[u]={}
	    for u2 in user_feature_avg:
                if u!=u2:
                    sim = 1 - spatial.distance.cosine(user_feature_avg[u], user_feature_avg[u2])
                    u_cosim[u][u2]=sim
	    u_cosim[u]=dict( sorted(u_cosim[u].items(),key=lambda item: item[1],reverse=True))  
	    
	# print(u_cosim)

	#get top five user for every user
	top_5={}
	top=5
	for u in u_cosim:
	    top_5[u]=[]
	    top_cur=0
	    for su in u_cosim[u]:
                top_5[u].append(su)
                top_cur+=1
                if top_cur==top:
                    break

	#print(top_5)

	#already watched content by a user
	watched_c=pd.DataFrame(columns=['user_id','watched_content'])
	for u in unique_user:
	    contents=list(r_data.loc[r_data['user_id']==u]['content_id'].unique())
	    watched_c.loc[len(watched_c)]=[u,contents]

	#similar users contents
	all_recom={}
	for u in top_5:
	    all_simu_content=[]
	    all_simu_content_unique=[]
	    all_not_watched=[]
	    simu=top_5[u]
	    for su in simu:
                su_content=watched_c.loc[watched_c['user_id']==su]['watched_content'].iloc[0]
                all_simu_content=all_simu_content+su_content
	    
	    for cont in all_simu_content:
                if cont not in all_simu_content_unique:
                    all_simu_content_unique.append(cont)   
	    
	    #remove already watched content
	    u_content=watched_c.loc[watched_c['user_id']==u]['watched_content'].iloc[0]
	    for cont in all_simu_content_unique:
                if cont not in u_content:
                    all_not_watched.append(cont)
		
	    all_recom[u]=all_not_watched
	    
	#store all recommendations
	recommendations=pd.DataFrame(columns=['user_id','content_recommendation'])
	for user in all_recom:
	    recommendations.loc[len(recommendations)]=[user,all_recom[user]]

	recommendations.to_csv('./all_user_recommendations.csv')

	#locality wise contents recommendation
	loc_content=pd.DataFrame(columns=['loc','content','freq'])
	#unique_locality
	u_loc=list(r_data['loc'].unique())
	cnt=r_data.groupby(['loc','content_id'])['content_id'].count()
	for c in cnt.iteritems():
	    loc_content.loc[len(loc_content)]=[c[0][0],c[0][1],c[1]]
	    
	locwise_recom=pd.DataFrame(columns=['location','content_recommendation'])
	for x in localities:
	    spec_loc_df=loc_content.loc[loc_content['loc']==x][['content','freq']]
	    spec_loc_df=spec_loc_df.sort_values('freq', ascending=False).reset_index(drop=True)
	    loc_spec_sorted_c=list(spec_loc_df['content'])
	    locwise_recom.loc[len(locwise_recom)]=[x,loc_spec_sorted_c]
	    
	locwise_recom.to_csv('./location_wise_recommendation.csv')

#get first common optimal content
def first_common_optimal_content(l1,l2):
	content_list=[]
	for item in l1:
	    if item in l2:
	        content_list.append(item)

	if len(content_list)==0:  #if no common content then show content after merging location and similarity suggestion
	    content_list=l1[:3]+l2[:2]

	return content_list

inp_file=pd.read_csv('./input_file_new_user.csv') # For old users give input_file.csv, for new user give input_file_new_user.csv
id_title=pd.read_csv('./id_to_title.csv')
loc_rec=pd.read_csv('./location_wise_recommendation.csv')

referesh_count=0
for index,row in inp_file.iterrows():
	u_id=row['user_id']
	if referesh_count<1 and row['new_old']==0: ## 0 for new user and 1 for old user
            print("New users have been found, we will add them in our algorithm before recommendations, it will take some time...")
            referesh_database()
            referesh_count+=1
	recomm_file=pd.read_csv('./all_user_recommendations.csv')
	recom=recomm_file.loc[recomm_file['user_id']==u_id,'content_recommendation'].iloc[0]
	if isinstance(recom,str):
	    recom=literal_eval(recom)
	    
	if row['loc'] not in [1,2,3,4,5,6,7]:  ## no locality is provided by user : overall recommendation
	    print("----- Recommendation for user: ",u_id,'without location preference -----')
	    for n in range(total_recom):
                try:
                    print(str(n+1)+':',id_title.loc[id_title['id']==recom[n],'title'].iloc[0])
                except:
                    pass
	else:
            print("----- Recommendation for user: ",u_id,'with location preference -----')
            loc_recom=loc_rec.loc[loc_rec['location']==row['loc'],'content_recommendation'].iloc[0]
            if isinstance(loc_recom,str):
                loc_recom=literal_eval(loc_recom)
            suggest=first_common_optimal_content(loc_recom,recom)
            for n in range(total_recom):
                try:
                    print(str(n+1)+':',id_title.loc[id_title['id']==suggest[n],'title'].iloc[0])
                except:
                    pass
        
