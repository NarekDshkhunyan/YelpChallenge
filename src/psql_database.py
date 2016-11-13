#!/usr/bin/python
# -*- coding: utf-8 -*-

import psycopg2
import sys
import json
import time

'''
Usage: 
    Make sure you have a psql instance running anc create a database called NLP_Yelp_Dataset
        - CREATE DB NLP_Yelp_Dataset
    make sure the relevant json files are in the topmost level of the folder yelp_data
    run this script ONLY ONCE!
    if you ran it before and want to rerun/ reset your tables, please make sure to DROP TABLE table_name
'''

con = None

try:
    # change these params to fit your set-up
    con = psycopg2.connect(database='NLP_Yelp_Dataset', user='Saadiyah') 
    cur = con.cursor()
    
    start = time.time()
    cur.execute("CREATE TABLE Business ( business_id varchar(25) PRIMARY KEY, name varchar(100), stars double precision, review_count INTEGER );")
    cur.execute("CREATE TABLE Users ( user_id varchar(25) PRIMARY KEY, review_count INTEGER, average_stars double precision, name varchar(100));")
    cur.execute("CREATE TABLE Review ( business_id varchar(25) references Business(business_id), user_id varchar(25) references Users(user_id), text varchar(5000), stars double precision, date DATE,vote_funny INTEGER , vote_useful INTEGER, vote_cool INTEGER);")
     
    # read in data from Business
    file_name="../yelp_data/yelp_academic_dataset_business.json"
    with open(file_name) as file:
        for line in file:
            entry = json.loads(line)
            business_id = entry["business_id"]
            name = entry["name"].encode('utf-8')
            name = name.replace("'", "''")
            stars = entry["stars"]
            review_count = entry["review_count"]
            query = "INSERT INTO Business VALUES('"+ str(business_id) +"','"+str(name) +"','"+ str(stars) +"',"+ str(review_count) +");"
            cur.execute(query);
     
    con.commit() #flush buffer
    
    print "time taken = ", time.time() - start
    start = time.time()
    
    # read in data from Users
    file_name="../yelp_data/yelp_academic_dataset_user.json"
    with open(file_name) as file:
        for line in file:
            entry = json.loads(line)
            user_id = entry["user_id"]
            review_count = entry["review_count"]
            average_stars = entry["average_stars"]
            name = entry["name"].encode('utf-8')
            name = name.replace("'", "''")
            query = "INSERT INTO users VALUES('"+ str(user_id) +"','"+str(review_count) +"','"+ str(average_stars) +"','"+ name +"');"
            cur.execute(query)
     
    con.commit() #flush buffer
    
    print "time taken = ", time.time() - start
    start = time.time()
    
    # read in data from Review
    file_name="../yelp_data/yelp_academic_dataset_review.json"
    
    count = 0
    with open(file_name) as file:
        for line in file:
            if count == 1000:
                count = 0
                con.commit()
            entry = json.loads(line)
            business_id = entry["business_id"]
            user_id = entry["user_id"]
            text = entry["text"].encode('utf-8')
            text = text.replace("'", "''")
            stars = entry["stars"]
            date = entry["date"]
            votes = entry["votes"]
            vote_funny = entry["votes"]["funny"]
            vote_useful = entry["votes"]["useful"]
            vote_cool = entry["votes"]["cool"] 
            query = "INSERT INTO review VALUES('"+ str(business_id) +"','"+str(user_id) +"','"+ str(text) +"','"+ str(stars) +"','"+ str(date) +"','"+ str(vote_funny) +"','"+ str(vote_useful) +"','"+ str(vote_cool) +"');"         
            cur.execute(query)
            count += 1
    
    con.commit()  #flush buffer
    
    print "time taken = ", time.time() - start
    
except psycopg2.DatabaseError, e:
    if con:
        con.rollback()
    
    print 'Error %s' % e    
    sys.exit(1)
    
finally:
    
    if con:
        con.close()
