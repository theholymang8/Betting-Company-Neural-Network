#imports
# coding=utf-8
import sqlite3
import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot as plt
#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#Filtraroume ta warnings gia na mhn emfanizontai kata thn ektelesh tou programmatos
warnings.filterwarnings('ignore')
#Syndeoume thn vash dedomenwn me to programma
conn = sqlite3.connect(r"C:\Users\30694\Desktop\Utilities\StudentProjects\5thSemester\PatternRecognition\PATTERN_REC_PROJECT\db.sqlite")

#Vazoume se metavlites twn pinakwn apo thn vash dedomenwn
dataset = pd.read_sql_query("SELECT * from MatchDropped", conn)
ScoreDB = pd.read_sql_query("SELECT * from Scores", conn)
TeamDB = pd.read_sql_query("SELECT * from Team_Attributes", conn)
B365c = pd.read_sql_query("SELECT * from B365Chance", conn)
Bwc= pd.read_sql_query("SELECT * from BWChance", conn)
Iwc = pd.read_sql_query("SELECT * from IWChance", conn)
Lbc = pd.read_sql_query("SELECT * from LBChance", conn)
WinDB = pd.read_sql_query("SELECT * from WinTable", conn)
NevrwnikoDB = pd.read_sql_query("SELECT * from NeuralTable", conn)
NevrwinikoMatchDB = pd.read_sql_query("SELECT * from NeuralMatches", conn)
NevrwnikoWinDB = pd.read_sql_query("SELECT * from NeuralWin", conn)



# Me thn synarthsh Chance(), vazoume ston pinaka ths kathe stoixhmatikhs etairias thn pithanothta poy dinei gia kathe omada na nikhsei
# ton kathe agwna. edo ginete gia thn bet365
def Chance():
    list_a = []
    list_b = []
    list_c = []

    list_end_a = []
    list_end_b = []
    list_end_c = []
    for x in dataset.B365H:
        list_a.append(x)
    for y in dataset.B365D:
        list_b.append(y)
    for z in dataset.B365A:
        list_c.append(z)
    for x in range(len(list_a)):
        sum = list_a[x] + list_b[x] + list_c[x]
        sA = list_a[x] / sum
        sB = list_b[x] / sum
        sC = list_c[x] / sum

        #pragmatiki pithanotita 1- tin apodosi
        sA = 1 - sA
        sB = 1 - sB
        sC = 1 - sC

        sumB = sA + sB + sC

        sA = sA / sumB
        sB = sB / sumB
        sC = sC / sumB

        list_end_a.append(sA)
        list_end_b.append(sB)
        list_end_c.append(sC)

    B365c.insert(1, "HomeTeamWinChance", list_end_a, True)
    B365c.insert(2, "DrawChance", list_end_b, True)
    B365c.insert(3, "AwayTeamWinChance", list_end_c, True)
    B365c.to_sql("B365Chance", conn, if_exists='replace', index=False)

    print(B365c)



# Me thn synarthsh Score(), vazoume tis plirofories pou tha xreiastoume sto pinaka ScoreDB
def Score():
    dfTable = dataset.home_team_api_id
    dfTable.to_sql("Scores", conn, if_exists="replace")
    ScoreDB.insert(2, "away_team_api_id", dataset.away_team_api_id, True)
    ScoreDB.to_sql("Scores", conn, if_exists='replace')
    ScoreDB.insert(2, "WhoWon", dataset.home_team_goal - dataset.away_team_goal, True)
    ScoreDB.to_sql("ScoresFull", conn, if_exists='replace', index=False)


# Me thn synarthsh WinD(),ftiaxnoume sthlh sto pinaka ScoreDB() pou me vash ta goal pou mphkan se kathe match mas kanei append to
# katallhlo gramma
def WinD():
    list = []
    for x in ScoreDB.WhoWon:
        if x > 0:
            list.append('H')
        elif x < 0:
            list.append("A")
        else:
            list.append("D")
    print(list)
    ScoreDB.insert(3, 'Win', list, True)
    ScoreDB.to_sql("Scores", conn, if_exists='replace', index=False)
    print(ScoreDB)




# Me thn synarthsh WinTrue(), ftiaxnoume ton pinaka WinDB kai vazoume sths treis sthles tou poios nikhse symfwna me ta goal pou mphkan
def WinTrue():
    list_a = []
    list_b = []
    list_c = []

    for x in ScoreDB.Win:
        if x == "H":
            list_a.append(1)
            list_b.append(0)
            list_c.append(0)
        elif x == "D":
            list_a.append(0)
            list_b.append(1)
            list_c.append(0)
        else:
            list_a.append(0)
            list_b.append(0)
            list_c.append(1)

    #fitaxnoume to pinaka WinTable
    WinDB.insert(1, "Home_Team_Win", list_a, True)
    WinDB.insert(2, "DrawChance", list_a, True)
    WinDB.insert(3, "Away_Team_Win", list_c, True)
    WinDB.to_sql("WinTable", conn, if_exists='replace', index=False)

    print(WinDB)


#Me thn synarthsh NevrwnikoDBs(), ftiaxnoume ton pinaka pou tha xrhsimopoihsoume gia ta machine learning montela mas sthn synexeia.
def NevrwnikoDBs():

    #ftiaxe pinaka tis b365c, first 3 collumns  b365c
    NevrwnikoDB = B365c
    del NevrwnikoDB["Win"]
    #kathe stihoimatiki ksehorista
    NevrwnikoDB.insert(3, "BWHomeTeamWinChance", Bwc.HomeTeamWinChance, True)
    NevrwnikoDB.insert(4, "BWDrawChance", Bwc.DrawChance, True)
    NevrwnikoDB.insert(5, "BWAwayTeamWinChance", Bwc.AwayTeamWinChance, True)

    NevrwnikoDB.insert(6, "IWHomeTeamWinChance", Iwc.HomeTeamWinChance, True)
    NevrwnikoDB.insert(7, "IWDrawChance", Iwc.DrawChance, True)
    NevrwnikoDB.insert(8, "IWAwayTeamWinChance", Iwc.AwayTeamWinChance, True)

    NevrwnikoDB.insert(9, "LBHomeTeamWinChance", Lbc.HomeTeamWinChance, True)
    NevrwnikoDB.insert(10, "LBDrawChance", Lbc.DrawChance, True)
    NevrwnikoDB.insert(11, "LBAwayTeamWinChance", Lbc.AwayTeamWinChance, True)

    NevrwnikoDB.to_sql("NeuralTable", conn, if_exists='replace', index=False)


    print(NevrwnikoDB)

    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    list_7 = []
    list_8 = []


    for x in range(len(NevrwinikoMatchDB.home_team_api_id)):#
        for y in range(len(TeamDB.team_api_id)):
            #checkaroume an i omada einai i idia apo tous dio pinakes kai tote pername ta xaraktiristika tis kathe omadas stis listes
            if NevrwinikoMatchDB.away_team_api_id[x] == TeamDB.team_api_id[y]:
                list_1.append(TeamDB.buildUpPlaySpeed[y])
                list_2.append(TeamDB.buildUpPlayPassing[y])
                list_3.append(TeamDB.chanceCreationPassing[y])
                list_4.append(TeamDB.chanceCreationCrossing[y])
                list_5.append(TeamDB.chanceCreationShooting[y])
                list_6.append(TeamDB.defencePressure[y])
                list_7.append(TeamDB.defenceAggression[y])
                list_8.append(TeamDB.defenceTeamWidth[y])
                print(len(list_1))
                break




    NevrwnikoDB.insert(20, "AWAYbuildUpPlaySpeed", list_1, True)
    NevrwnikoDB.insert(21, "AWAYbuildUpPlayPassing", list_2, True)
    NevrwnikoDB.insert(22, "AWAYchanceCreationPassing", list_3, True)
    NevrwnikoDB.insert(23, "AWAYchanceCreationCrossing", list_4, True)
    NevrwnikoDB.insert(24, "AWAYchanceCreationShooting", list_5, True)
    NevrwnikoDB.insert(25, "AWAYdefencePressure", list_6, True)
    NevrwnikoDB.insert(26, "AWAYdefenceAggression", list_7, True)
    NevrwnikoDB.insert(27, "AWAYdefenceTeamWidth", list_8, True)



    NevrwnikoDB.to_sql("NeuralTable", conn, if_exists='replace', index = False)
    print(NevrwnikoDB)







# Me thn synarthsh MeanSquaredError(), vazoume tis pithanothtes pou vrikame oti dinei h kathe stoixhmatikh etairia kai tis sigkrinoume
# me tis pragmatikes.
def MeanSquaredError():

    xtrue = [WinDB.Home_Team_Win, WinDB.DrawChance, WinDB.Away_Team_Win]
    ypred = [Iwc.HomeTeamWinChance, Iwc.DrawChance, Iwc.AwayTeamWinChance]
    IW = np.square(np.subtract(xtrue, ypred)).mean()

    xtrue = [WinDB.Home_Team_Win, WinDB.DrawChance, WinDB.Away_Team_Win]
    ypred = [Lbc.HomeTeamWinChance, Lbc.DrawChance, Lbc.AwayTeamWinChance]
    LB = np.square(np.subtract(xtrue, ypred)).mean()

    xtrue = [WinDB.Home_Team_Win, WinDB.DrawChance, WinDB.Away_Team_Win]
    ypred = [B365c.HomeTeamWinChance, B365c.DrawChance, B365c.AwayTeamWinChance]
    B365 = np.square(np.subtract(xtrue, ypred)).mean()

    xtrue = [WinDB.Home_Team_Win, WinDB.DrawChance, WinDB.Away_Team_Win]
    ypred = [Bwc.HomeTeamWinChance, Bwc.DrawChance, Bwc.AwayTeamWinChance]
    BW = np.square(np.subtract(xtrue, ypred)).mean()



    print("IW   " + str(IW))
    print("LB   " + str(LB))
    print("B365 " + str(B365))
    print("BW   " + str(BW))
    print("B365 has the lowest mean squared error: " + str(B365))




# Me thn synarthsh LeastSquares(),efarmozoume thn methodo leastsquares kai vgazoume to error gia kathe agwna
# me to root squared error.An den theloume to meso tote arkei na valoume tis 2 teleftaies grammes tou kwdika se comment.
# Sthn sygkekrimenh synarthsh kanoume xrhsh ths Bw betting company, gia na doume ta apotelesmata gia kapoia allh prepei na
# allaxtei manually.
def LeastSquares():
    x =  [WinDB.Home_Team_Win, WinDB.DrawChance, WinDB.Away_Team_Win]
    y = [Bwc.HomeTeamWinChance, Bwc.DrawChance, Bwc.AwayTeamWinChance]


    # Vriskoume ta mesa
    x_mean = np.mean(x)
    y_mean = np.mean(y)


    # O arithos twn stoixeiwn
    n = len(x)

    #Xrhesimopoioume thn formula gia na vroume ta m,c
    numer = 0
    denom = 0
    for i in range(n):
        numer += (x[i] - x_mean) * (y[i] - y_mean)
        denom += (x[i] - x_mean) ** 2
    m = numer / denom #klisi eutheias
    c = y_mean - (m * x_mean) #simeio tomis

    #kanoume print ta coefficients(syntelestes) an den theloume ta bazoume se sxolia
    print ("Coefficients")
    print(m,c)



    #Dhmiourgoume tis metavlites gia na kanoume plot
    max_x = np.max(x) +100
    min_x = np.min(x) -100

    x = np.linspace(min_x,max_x, 22467)
    y = c+m*x

    # Plotting line
    plt.plot(x, y, color='#58b970', label='Regression Line')
    # Ploting Scatter Points
    plt.scatter(x, y, c='#ef5423', label='Scatter Plot')

    plt.xlabel('True Values')
    plt.ylabel('Betting Company Chance given')
    plt.legend()
    plt.show()

    #Ypologizoume to Root squared Error
    rmse = 0
    for i in range(n):
        y_pred = c+m*x[i]
        rmse += (y[i]-y_pred) **2
    rmse = np.sqrt(rmse/n)
    print("RMSE" )
    print (rmse)

    #An den theloume ta mesa bazoume autes tis grammes se sholia
    rmse = np.mean(rmse)
    print (rmse)





#Me thn synarthsh MachineLearning(), efarmozoume ta machine learning montela kai emfanizoume ta apotelesmata tous.
def MachineLearning():
    num_trees = 100
    test_size = 0.30
    seed      = 9
    scoring    = "accuracy" #parametros
    results = []
    names = []
    X =np.array([NevrwnikoDB.B365HomeTeamWinChance, NevrwnikoDB.B365DrawChance,NevrwnikoDB.B365AwayTeamWinChance,
                 NevrwnikoDB.BWHomeTeamWinChance,NevrwnikoDB.BWDrawChance, NevrwnikoDB.BWAwayTeamWinChance,
                NevrwnikoDB.IWHomeTeamWinChance, NevrwnikoDB.IWDrawChance, NevrwnikoDB.IWAwayTeamWinChance,
                NevrwnikoDB.LBHomeTeamWinChance, NevrwnikoDB.LBDrawChance, NevrwnikoDB.LBAwayTeamWinChance,
                 NevrwnikoDB.HOMEbuildUpPlaySpeed, NevrwnikoDB.HOMEbuildUpPlayPassing, NevrwnikoDB.HOMEchanceCreationPassing,
                 NevrwnikoDB.HOMEchanceCreationCrossing, NevrwnikoDB.HOMEchanceCreationShooting, NevrwnikoDB.HOMEdefencePressure,
                 NevrwnikoDB.HOMEdefenceAggression, NevrwnikoDB.HOMEdefenceTeamWidth,
                 NevrwnikoDB.AWAYbuildUpPlaySpeed,
                 NevrwnikoDB.AWAYbuildUpPlayPassing, NevrwnikoDB.AWAYchanceCreationPassing,
                 NevrwnikoDB.AWAYchanceCreationCrossing, NevrwnikoDB.AWAYchanceCreationShooting, NevrwnikoDB.AWAYdefencePressure,
                 NevrwnikoDB.AWAYdefenceAggression, NevrwnikoDB.AWAYdefenceTeamWidth])

    X = X.transpose()#metathesi stihoeion oste na ta katalavei san values
    Y = np.array(NevrwnikoWinDB.WIN)#pragmatikes ekbaseis

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=100)#stiheia train test


    models = []
    #O Logistic Regression eihe ena thema kai den emfanize data, mporei se allo systima na doulepesei giauto ton exoyme se sholia
    #models.append(('Logistic Regression', LogisticRegression(random_state=seed)))
    models.append(('Decision Tree', DecisionTreeClassifier(random_state=seed)))
    models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
    models.append(('Support Vector Machine', SVC(random_state=seed)))
    models.append(('Randοm Forest', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('Gaussian NB', GaussianNB()))
    models.append(('K Neighbours Classifier', KNeighborsClassifier()))


    print("Train data  : {}".format(X_train.shape))
    print("Test data   : {}".format(X_test.shape))
    print("Train labels: {}".format(Y_train.shape))
    print("Test labels : {}".format(Y_test.shape))


    for name, model in models: #10 fores taxinomisi
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)



if __name__ == "__main__":
    print("FIRST TASK: ")
    MeanSquaredError()
    print("SECOND TASK")
    LeastSquares()
    print("THIRD TASK")
    MachineLearning()














