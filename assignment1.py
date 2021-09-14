# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import csv
import re
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'
with open(datafilepath, encoding="utf8") as json_file:
        data = json.load(json_file)

def task1():
    #Complete task 1 here

    return sorted([code['club_code'] for code in data['clubs']])
    
def task2():
    #Complete task 2 here
    GoalsPerTeam = [[team['club_code'], team['goals_scored'], team['goals_conceded']] for team in data['clubs']]
    GoalsPerTeam = sorted(GoalsPerTeam, key=lambda x: x[0])

    with open ('task2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['team', 'goals scored','goals conceded'])
        writer.writerows(GoalsPerTeam)
    return
      
def task3():
    #Complete task 3 here
    ArticleMatches = []
    for filenum in range(1,266):
        filename = f'{filenum:03}' + '.txt'
        filepath = articlespath + '/' + filename

        with open(filepath, 'r', encoding='utf8') as currentarticle:
            text = currentarticle.read()
            matches = re.findall("(\d+(?<!\d{3})-\d+)",text)

            scores = [int(tally.split('-')[0])+int(tally.split('-')[1]) for tally in matches]

            if len(scores) == 0:
                ArticleMatches.append([filename,0])
            else:
                ArticleMatches.append([filename, max(scores)])

    with open ('task3.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'max tally'])
        writer.writerows(ArticleMatches)
    return

def task4():
    #Complete task 4 here
    task3()
    with open('task3.csv', 'r', encoding='utf8') as TotalGoals:
        next(TotalGoals)
        data = [item[1] for item in TotalGoals]
        data = [int(datum) for datum in data]

    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)
    plt.savefig('task4.png')

    return
    
def task5():
    #Complete task 5 here
    Teams = data['participating_clubs']
    TeamMentions = {i:0 for i in Teams}

    for filenum in range(1,266):
        filename = f'{filenum:03}' + '.txt'
        filepath = articlespath + '/' + filename

        with open(filepath, 'r', encoding='utf8') as currentarticle:
            text = currentarticle.read()
            for team in Teams:
                if text.count(team) > 0:
                    TeamMentions[team] += 1

    with open ('task5.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['team', 'mentions'])
            writer.writerows(sorted(TeamMentions.items(), key=lambda x:x[0]))
    keys = TeamMentions.keys()
    values = TeamMentions.values()

    plt.title("Team Mentions")
    plt.bar(keys, values)
    plt.xticks(rotation=70)
    plt.xlabel("Team")
    plt.ylabel("Article Mentions")
    plt.savefig("task5.png",bbox_inches="tight")
    return

def task6():
    #Complete task 6 here
    with open(datafilepath, encoding="utf8") as json_file:
        data = json.load(json_file)
        Teams = data['participating_clubs']
    task5()
    plt.close()

    with open('task5.csv', 'r', encoding='utf8') as data:
        next(data)
        TeamMentions = {line.strip('\n').split(",")[0]: int(line.strip('\n').split(",")[1]) for line in data}

        ArticleMentions = {team:set() for team in Teams}
        #create dict where article titles are the keys and the items are the teams mentioned
        for filenum in range(1,266):
            filename = f'{filenum:03}' + '.txt'
            filepath = articlespath + '/' + filename

            with open(filepath, 'r', encoding='utf8') as currentarticle:
                text = currentarticle.read()
                for team in Teams:
                    if team in text:
                        ArticleMentions[team].add(filename)

    #Calculate the Similarity Score
    SimilarityScores = [[0 for i in range(len(Teams))] for x in range(len(Teams))]
    for i in range(len(Teams)):
        for j in range(len(Teams)):
            TeamA = Teams[i]
            TeamB = Teams[j]
            CommonMentions = ArticleMentions[TeamA].intersection(ArticleMentions[TeamB])
            if TeamA == TeamB:
                SimilarityScores[i][j] = 1
            else:
                SimilarityScores[i][j] = (2*len(CommonMentions))/(TeamMentions[TeamA] + TeamMentions[TeamB])

    SimilarityScores = np.array(SimilarityScores)        

    #Create the Plot
    mask = np.zeros_like(SimilarityScores)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(SimilarityScores, mask=mask, vmax=.3, square=True,  cmap="YlGnBu")
        ax.set_title("Heatmap: Team Article Mentions Similarity")
        ax.set_xlabel("Teams")
        ax.set_ylabel("Teams")
        ax.set_xticklabels(Teams)
        ax.set_yticklabels(Teams)
        plt.xticks(rotation=70)
        plt.yticks(rotation=360)
        plt.savefig('task6.png',bbox_inches="tight")
    return
    
def task7():
    #Complete task 7 here
    task2()
    task5()
    plt.close()

    with open('task2.csv', 'r', encoding='utf8') as data:
        next(data)
        GoalScored = {line.strip('\n').split(",")[0]: int(line.strip('\n').split(",")[1]) for line in data}
    with open('task5.csv', 'r', encoding='utf8') as data:
        next(data)
        TeamMentions = {line.strip('\n').split(",")[0]: int(line.strip('\n').split(",")[1]) for line in data}

    GoalsScored_values = GoalScored.values()
    TeamMentions_values = TeamMentions.values()

    plt.scatter(TeamMentions_values,GoalsScored_values)
    plt.title("Relationship Between Article Mentions and Total Goal Scores")
    plt.xlabel("Articles Mentioning Each Team")
    plt.ylabel("Total Goals Scored per Team")
    plt.savefig("task7.png",bbox_inches="tight")
    return

    
def task8(filename):
    #Complete task 8 here
    filepath = filename
    with open(filepath, 'r', encoding='utf8') as CurrentArticle:
        text = CurrentArticle.read()
        StopWords = set(stopwords.words('english'))
        text = re.sub(r'[^\w\s]',' ',text)   #remove punctuations
        textTokenized = word_tokenize(text) #tokenize the text
        textTokenized = [word.lower() for word in textTokenized if word.isalpha() and len(word) != 1]
        textTokenized = [word for word in textTokenized if word not in StopWords]

    return textTokenized

