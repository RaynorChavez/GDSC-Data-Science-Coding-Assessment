# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
from PIL.Image import ROTATE_90
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import re
import csv
import seaborn as sns
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

with open(datafilepath, encoding="utf8") as json_file:
    data = json.load(json_file)

#task1
#print(sorted([code['club_code'] for code in data['clubs']]))

#task2
#GoalsPerTeam = [[team['club_code'], team['goals_scored'], team['goals_conceded']] for team in data['clubs']]
#print(sorted(GoalsPerTeam, key=lambda x: x[0]))

#task3
'''
ArticleMatches = []
for filenum in range(1,266):
    filename = f'{filenum:03}' + '.txt'
    filepath = articlespath + '/' + filename

    with open(filepath, 'r', encoding='utf8') as currentarticle:
        text = currentarticle.read()
        #matches = re.findall("(\d+-\d+)",text)
        matches = re.findall("(\d+(?<!\d{3})-\d+)",text)
        #print(matches)

        scores = [int(tally.split('-')[0])+int(tally.split('-')[1]) for tally in matches]
        for score in scores:
            if score > 1000:
                scores.remove(score)

        if len(scores) == 0:
            ArticleMatches.append([filename,0])
        else:
            ArticleMatches.append([filename, max(scores)])
        #print("\n")

for i in range(1,266):
    print(str(ArticleMatches[i]) + "--" + str(answer[i]) + "\n")
with open ('task3.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'max tally'])
        writer.writerows(ArticleMatches)
'''
#Task 4
'''
with open('task3.csv', 'r', encoding='utf8') as TotalGoals:
    next(TotalGoals)
    data = [item[1] for item in TotalGoals]

data = [int(datum) for datum in data]

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(data)
plt.savefig('task4.png')
'''

#Task 5
'''
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
plt.savefig("task5.png")
'''
#Task 6
'''Teams = data['participating_clubs']
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
            #ArticleMentions[filename] = set(team for team in Teams if team in text)

SimilarityScores = [[0 for i in range(len(Teams))] for x in range(len(Teams))]
for i in range(len(Teams)):
    for j in range(len(Teams)):
        TeamA = Teams[i]
        TeamB = Teams[j]
        CommonMentions = ArticleMentions[TeamA].intersection(ArticleMentions[TeamB])
        #print(TeamA, TeamB, len(CommonMentions), "\n")
        if TeamA == TeamB:
            SimilarityScores[i][j] = 1
        else:
            SimilarityScores[i][j] = (2*len(CommonMentions))/(TeamMentions[TeamA] + TeamMentions[TeamB])

SimilarityScores = np.array(SimilarityScores)        


mask = np.zeros_like(SimilarityScores)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(SimilarityScores, mask=mask, vmax=.3, square=True,  cmap="YlGnBu")
    plt.title("Heatmap: Team Article Mentions Similarity")
    ax.set_xlabel("Teams")
    ax.set_ylabel("Teams")
    ax.set_xticklabels(Teams)
    ax.set_yticklabels(Teams)
    plt.xticks(rotation=70)
    plt.yticks(rotation=360)
    plt.savefig('task6.png')'''

#Task 7
'''with open('task2.csv', 'r', encoding='utf8') as data:
    next(data)
    GoalScored = {line.strip('\n').split(",")[0]: int(line.strip('\n').split(",")[1]) for line in data}
with open('task5.csv', 'r', encoding='utf8') as data:
    next(data)
    TeamMentions = {line.strip('\n').split(",")[0]: int(line.strip('\n').split(",")[1]) for line in data}
print(GoalScored)
print(TeamMentions)

GoalsScored_keys = GoalScored.keys()
GoalsScored_values = GoalScored.values()
TeamMentions_keys = TeamMentions.keys()
TeamMentions_values = TeamMentions.values()
plt.scatter(TeamMentions_values,GoalsScored_values)
plt.title("Relationship Between Article Mentions and Total Goal Scores")
plt.xlabel("Articles Mentioning Each Team")
plt.ylabel("Total Goals Scored per Team")
plt.savefig("task7.png",bbox_inches="tight")'''

#Task 8
'''
== Remove all non-alphabetic characters (for example, numbers, apostrophes and punctuation characters), 
except for spacing characters such as whitespaces, tabs and newlines.
== Convert all spacing characters such as tabs and newlines to whitespace and ensure that only one
whitespace character exists between each word
== Change all uppercase characters to lower case
== Tokenize the resulting string into words
== Remove all stopwords in nltk's (Natural Language Toolkit Python lib) list of English stopwords from the resulting list
== Remove all remaining words that are only a single character long from the resulting list
'''

filename = '001.txt'
filepath = articlespath + "/"+ filename
with open(filepath, 'r', encoding='utf8') as CurrentArticle:
    text = CurrentArticle.read()
    StopWords = set(stopwords.words('english'))
    #text = text.lower()                 #lowercase
    #text = re.sub(r'\d+','',text)       #remove numbers
    text = re.sub(r'[^\w\s]',' ',text)   #remove punctuations
    #text = text.strip()                 #converting spacing characters
    textTokenized = word_tokenize(text) #tokenize the text
    textTokenized = [word.lower() for word in textTokenized if word.isalpha() and len(word) != 1]
    textTokenized = [word for word in textTokenized if word not in StopWords]
    #
    #textTokenized = [word.lower() for word in textTokenized if word not in StopWords and len(word) != 1]   #remove stopwords and single char words

answer=['man', 'utd', 'stroll', 'cup', 'win', 'wayne', 'rooney', 'made', 'winning', 'return', 'everton', 'manchester', 'united', 'cruised', 'fa', 'cup', 'quarter', 'finals', 'rooney', 'received', 'hostile', 'reception', 'goals', 'half', 'quinton', 'fortune', 'cristiano', 'ronaldo', 'silenced', 'jeers', 'goodison', 'park', 'fortune', 'headed', 'home', 'minutes', 'ronaldo', 'scored', 'nigel', 'martyn', 'parried', 'paul', 'scholes', 'free', 'kick', 'marcus', 'bent', 'missed', 'everton', 'best', 'chance', 'roy', 'carroll', 'later', 'struck', 'missile', 'saved', 'feet', 'rooney', 'return', 'always', 'going', 'potential', 'flashpoint', 'involved', 'angry', 'exchange', 'spectator', 'even', 'kick', 'rooney', 'every', 'touch', 'met', 'deafening', 'chorus', 'jeers', 'crowd', 'idolised', 'year', 'old', 'everton', 'started', 'brightly', 'fortune', 'needed', 'alert', 'scramble', 'away', 'header', 'bent', 'near', 'goal', 'line', 'cue', 'united', 'take', 'complete', 'control', 'supreme', 'passing', 'display', 'goodison', 'park', 'pitch', 'cutting', 'fortune', 'gave', 'united', 'lead', 'minutes', 'rising', 'meet', 'ronaldo', 'cross', 'eight', 'yards', 'portuguese', 'youngster', 'allowed', 'much', 'time', 'space', 'hapless', 'gary', 'naysmith', 'united', 'dominated', 'without', 'creating', 'many', 'clear', 'cut', 'chances', 'almost', 'paid', 'price', 'making', 'domination', 'two', 'minutes', 'half', 'time', 'mikel', 'arteta', 'played', 'superb', 'ball', 'area', 'bent', 'played', 'onside', 'gabriel', 'heintze', 'hesitated', 'carroll', 'plunged', 'fee', 'save', 'united', 'almost', 'doubled', 'lead', 'minutes', 'ronaldo', 'low', 'drive', 'yards', 'took', 'deflection', 'tony', 'hibbert', 'martyn', 'dived', 'save', 'brilliantly', 'martyn', 'came', 'everton', 'rescue', 'three', 'minutes', 'later', 'rooney', 'big', 'moment', 'almost', 'arrived', 'raced', 'clean', 'veteran', 'keeper', 'outstanding', 'form', 'nothing', 'martyn', 'could', 'united', 'doubled', 'lead', 'minutes', 'doubled', 'advantage', 'scholes', 'free', 'kick', 'took', 'deflection', 'martyn', 'could', 'parry', 'ball', 'ronaldo', 'reacted', 'first', 'score', 'easily', 'everton', 'problems', 'worsened', 'james', 'mcfadden', 'limped', 'injury', 'may', 'trouble', 'ahead', 'everton', 'goalkeeper', 'carroll', 'required', 'treatment', 'struck', 'head', 'missile', 'thrown', 'behind', 'goal', 'rooney', 'desperate', 'search', 'goal', 'return', 'everton', 'halted', 'martyn', 'injury', 'time', 'outpaced', 'stubbs', 'martyn', 'denied', 'england', 'striker', 'manchester', 'united', 'coach', 'sir', 'alex', 'ferguson', 'fantastic', 'performance', 'us', 'fairness', 'think', 'everton', 'missed', 'couple', 'players', 'got', 'young', 'players', 'boy', 'ronaldo', 'fantastic', 'player', 'persistent', 'never', 'gives', 'know', 'many', 'fouls', 'gets', 'wants', 'ball', 'truly', 'fabulous', 'player', 'everton', 'martyn', 'hibbert', 'yobo', 'stubbs', 'naysmith', 'osman', 'carsley', 'arteta', 'kilbane', 'mcfadden', 'bent', 'subs', 'wright', 'pistone', 'weir', 'plessis', 'vaughan', 'manchester', 'united', 'carroll', 'gary', 'neville', 'brown', 'ferdinand', 'heinze', 'ronaldo', 'phil', 'neville', 'keane', 'scholes', 'fortune', 'rooney', 'subs', 'howard', 'giggs', 'smith', 'miller', 'spector', 'referee', 'styles', 'hampshire']

print(len(textTokenized), len(answer))
diff =[]
for word in textTokenized:
    if word not in answer:
        diff.append(word)

for word in answer:
    if word not in textTokenized:
        diff.append(word)
print(diff)

for i in range(len(textTokenized)):
    print(textTokenized[i], answer[i], "\n")



