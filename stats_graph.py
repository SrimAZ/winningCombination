import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pulp
import re
import sys


def gengraph(pname, tname):

    pname = pname.strip()
    fdn = pd.read_csv("bowler_name_mvpi.csv")
    fdn1 = fdn.set_index("Player_Name", drop=False)
    binfo = fdn1.loc[pname]
    bid = binfo['Player_Id']

    ma = pd.read_csv('Match.csv')
    df = pd.read_csv('Ball_by_Ball.csv')
    ma = pd.read_csv('Match.csv')
    df['Batsman_Scored'].replace('Do_nothing', '0', inplace=True)
    df['Batsman_Scored'].replace(' ', '0', inplace=True)
    df['Extra_Runs'].replace(' ', '0', inplace=True)
    df.Batsman_Scored = pd.to_numeric(df['Batsman_Scored'], errors='coerse')
    df.Batsman_Scored = pd.to_numeric(df['Batsman_Scored'], errors='coerse')
    df.Extra_Runs = pd.to_numeric(df['Extra_Runs'], errors='coerse')

    df['Dissimal_Type'].replace(['bowled', 'caught', 'hit wicket', 'stumped', 'caught and bowled', 'lbw'], '1',
                                inplace=True)
    df['Dissimal_Type'].replace([' ', 'run out', 'retired hurt', 'obstructing the filed'], '0', inplace=True)
    df.Dissimal_Type = pd.to_numeric(df['Dissimal_Type'], errors='coerse')
    df['No_Balls'] = 1
    wick_match = df.groupby(['Match_Id', 'Bowler_Id'])[['Dissimal_Type', 'No_Balls']].sum().reset_index()
    wick_match['Bowler_Strike_Rate'] = wick_match['No_Balls'] / wick_match['Dissimal_Type']
    wick_match.columns = ['Match_Id', 'Bowler_Id', 'No_Of_Wickets_Match', 'No_Of_Balls_Match',
                          'Bowler_Strike_Rate_Match']
    # wick=wick_match.groupby(['Match_Id'])[['No_Of_Wickets_Match']].sum().reset_index()
    wick = wick_match[wick_match['Bowler_Id'] == bid]
    runsconc_match = df.groupby(['Match_Id', 'Bowler_Id'])[['Batsman_Scored', 'No_Balls']].sum().reset_index()
    runsconc_match.columns = ['Match_Id', 'Bowler_Id', 'Runs_Conceeded_Match', 'No_Balls']
    runsconc_match['Economy_Rate_Match'] = runsconc_match['Runs_Conceeded_Match'] / (runsconc_match['No_Balls'] / 6)
    runsconc_match = runsconc_match[runsconc_match['Bowler_Id'] == bid]

    d = df[(df['Over_Id'] >= 15) & (df['Batsman_Scored'] == 0)][
        ['Over_Id', 'Match_Id', 'Bowler_Id', 'Batsman_Scored', 'Team_Bowling_Id']]
    d['Batsman_Scored'].replace(0, 1, inplace=True)
    ldotballs = d.groupby(['Match_Id', 'Bowler_Id', 'Team_Bowling_Id'])[['Batsman_Scored']].sum().reset_index()
    ldotballs.columns = ['Match_Id', 'Bowler_Id', 'Team_Bowling_Id', 'Last_Dot_Balls']
    ldotballs = ldotballs[ldotballs['Bowler_Id'] == bid]

    fi, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fi.suptitle('Bowler Evaluation')

    g = wick['No_Of_Wickets_Match'].values
    ax1.plot(g)
    ax1.set_title("Wickets in each Match")
    ax1.set_ylabel('Wickets')

    e = runsconc_match['Runs_Conceeded_Match'].values
    ax2.plot(e, color='orange')
    ax2.set_title("'Runs Conceeded in each Match")
    ax2.set_ylabel('Runs')

    # w=f[f['Team_Batting_Id']==3]
    g = runsconc_match['Economy_Rate_Match'].values
    ax3.plot(g, color='red')
    ax3.set_title("Economy Rate in each Match")
    ax3.set_ylabel('Economy Rate')

    # w=f[f['Team_Batting_Id']==4]
    g = ldotballs['Last_Dot_Balls'].values
    ax4.plot(g, color='green')
    ax4.set_title("Dot Balls Death Over in  Matches")
    ax4.set_ylabel('Number of Dot Balls')
    plt.show()

    return "Graph has been generated for %s"%pname

def genbatgraph(pname, tname):

    pname = pname.strip()
    fdn = pd.read_csv("batsman_name_mvpi.csv")
    fdn1 = fdn.set_index("Player_Name", drop=False)
    binfo = fdn1.loc[pname]
    bid = binfo['Player_Id']

    ma = pd.read_csv('Match.csv')
    df = pd.read_csv('Ball_by_Ball.csv')
    ma = pd.read_csv('Match.csv')
    df['Batsman_Scored'].replace('Do_nothing', '0', inplace=True)
    df['Batsman_Scored'].replace(' ', '0', inplace=True)
    df['Extra_Runs'].replace(' ', '0', inplace=True)
    df.Batsman_Scored = pd.to_numeric(df['Batsman_Scored'], errors='coerse')
    df.Batsman_Scored = pd.to_numeric(df['Batsman_Scored'], errors='coerse')
    df.Extra_Runs = pd.to_numeric(df['Extra_Runs'], errors='coerse')

    df['No_Balls'] = 1
    runs_match = df.groupby(['Match_Id', 'Striker_Id'])[['Batsman_Scored', 'No_Balls']].sum().reset_index()
    runs_match['Strike_Rate_Match'] = runs_match['Batsman_Scored'] / runs_match['No_Balls'] * 100
    runs_match.columns = ['Match_Id', 'Striker_Id', 'Batsman_Scored_Match', 'No_Balls_Faced_Match', 'Strike_Rate_Match']
    runs_match = runs_match[runs_match['Striker_Id'] == bid]

    d = df[df['Over_Id'] <= 6][['Over_Id', 'Match_Id', 'Striker_Id', 'Batsman_Scored']]
    fruns = d.groupby(['Match_Id', 'Striker_Id'])[['Batsman_Scored']].sum().reset_index()
    # di=fruns.groupby(['Striker_Id'])[['Batsman_Scored']].sum().reset_index()
    # di=di[di['Striker_Id']==1]
    fruns = fruns[fruns['Striker_Id'] == bid]
    print fruns.head()
    # # print di.head()

    d = df[(df['Over_Id'] <= 6) & (df['Batsman_Scored'] == 6)][['Over_Id', 'Match_Id', 'Striker_Id', 'Batsman_Scored']]
    d['Batsman_Scored'].replace(6, 1, inplace=True)
    fsix = d.groupby(['Match_Id', 'Striker_Id'])[['Batsman_Scored']].sum().reset_index()
    fsix.columns = ['Match_Id', 'Striker_Id', 'First_Sixes']
    fsix = fsix[fsix['Striker_Id'] == bid]
    # print fsix.head()

    d = df[(df['Over_Id'] >= 15) & (df['Batsman_Scored'] == 4)][['Over_Id', 'Match_Id', 'Striker_Id', 'Batsman_Scored']]
    d['Batsman_Scored'].replace(4, 1, inplace=True)
    lfour = d.groupby(['Match_Id', 'Striker_Id'])[['Batsman_Scored']].sum().reset_index()
    lfour.columns = ['Match_Id', 'Striker_Id', 'Last_Fours']
    lfour = lfour[lfour['Striker_Id'] == bid]

    fi, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fi.suptitle('Batsman Evaluation')

    # # w=f[f['Team_Batting_Id']==1]
    g = runs_match['Strike_Rate_Match'].values
    ax1.plot(g)
    ax1.set_title("Strike Rate in Each Match")
    ax1.set_ylabel('Strike Rate')
    ax1.set_xlabel('Match')

    # # d=f[f['Team_Batting_Id']==2]
    e = fruns['Batsman_Scored'].values
    ax2.plot(e, color='orange')
    ax2.set_title("Runs in Power Play")
    ax2.set_ylabel('Runs')
    ax2.set_xlabel('Match')

    # plt.xlim(-5,25)
    # # plt.ylim(0,10)

    # # boxplot(g['No_of_Fours'].values,positions=[1],widths=10)
    # ax2.set_title("'Runs Conceeded in each Match")
    # # ax2.plot
    # ax2.set_ylabel('Runs')

    # # w=f[f['Team_Batting_Id']==3]
    g = fsix['First_Sixes'].values
    ax3.plot(g, color='red')
    ax3.set_title("Number of Sixes in Power Play")
    ax3.set_ylabel('Number of Sixes')
    ax3.set_xlabel('Match')

    # w=f[f['Team_Batting_Id']==4]
    g = lfour['Last_Fours'].values
    ax4.plot(g, color='green')
    ax4.set_title("Fours in Death Over")
    ax4.set_ylabel('Number of Fours')
    ax4.set_xlabel('Match')

    plt.show()
    return "Graph has been generated for %s" % pname

def genteam(nob,nobl,bat_budget,bowl_budget,spin_no):

    bat = pd.read_csv('optibats.csv')
    sys.setrecursionlimit(2500)
    bat['Base_Price'].replace(' ', '0', inplace=True)
    bat.Base_Price = pd.to_numeric(bat['Base_Price'], errors='coerse')
    bowl = pd.read_csv('optibowl.csv')
    bat['Batsman'] = 1

    problem = pulp.LpProblem('IPLAuction', pulp.LpMaximize)

    decision_variables = []
    for rownumber, row in bat.iterrows():
        v = str('x' + str(rownumber))
        v = pulp.LpVariable(str(v), lowBound=0, upBound=1, cat='Integer')
        decision_variables.append(v)

    # maximization function
    player_rating = ""
    for rownum, row in bat.iterrows():
        for i, j in enumerate(decision_variables):
            if rownum == i:
                f = row['Batsman_Index'] * j
                player_rating += f
    # print decision_variables
    problem += player_rating
    bat = bat.fillna(9999)

    budget = int(bat_budget)
    cur_cost = ""
    for rownum, row in bat.iterrows():
        for i, j in enumerate(decision_variables):
            if rownum == i:
                # print row['Base_Price']
                f = row['Base_Price'] * j
                # print f
                cur_cost += f
    # print cur_cost
    problem += (cur_cost <= budget )

    no_batsm = int(nob)

    cur_batsm = ""
    for rownum, row in bat.iterrows():
        for i, j in enumerate(decision_variables):
            if rownum == i:
                # print row['Base_Price']
                f = row['Batsman'] * j
                # print f
                cur_batsm += f
    # print cur_cost
    problem += (no_batsm == cur_batsm)
    optimization_result = problem.solve()

    assert optimization_result == pulp.LpStatusOptimal

    variable_name = []
    variable_value = []

    for v in problem.variables():
        variable_name.append(v.name)
        variable_value.append(v.varValue)

    df = pd.DataFrame({'variable': variable_name, 'value': variable_value})
    for rownum, row in df.iterrows():
        value = re.findall(r'(\d+)', row['variable'])
        df.loc[rownum, 'variable'] = int(value[0])

    df = df.sort_index(by='variable')

    # append results
    for rownum, row in bat.iterrows():
        for results_rownum, results_row in df.iterrows():
            if rownum == results_row['variable']:
                bat.loc[rownum, 'decision'] = results_row['value']

    batdf = (bat[bat['decision'] == 1][['Player_Name', 'Country', 'Batsman_Index']])
    batdf.to_csv("batsman_selected.csv")

    # Spinners

    bowl = pd.read_csv('optibowl.csv')
    bowl['Bowler_price'].replace(' ', '0', inplace=True)
    bowl.Base_Price = pd.to_numeric(bowl['Bowler_price'], errors='coerse')
    pla = pd.read_csv('Player.csv')
    bowl = pd.merge(bowl, pla, on=['Player_Id', 'Player_Name'], how='left')[
        ['Player_Id', 'Player_Name', 'Bowler_price', 'Bowler_Index', 'Bowling_Skill']]
    bowl['Bowler_price'] = bowl['Bowler_price'] / 100
    bowl = bowl.fillna(9999)
    bd = bowl[(bowl['Bowling_Skill'] == 'Right-arm medium') | (bowl['Bowling_Skill'] == 'Right-arm fast-medium') | (
            bowl['Bowling_Skill'] == 'Right-arm medium-fast') | (bowl['Bowling_Skill'] == 'Right-arm fast') | (
                      bowl['Bowling_Skill'] == 'Right-arm bowler') | (
                      bowl['Bowling_Skill'] == 'Left-arm fast-medium') | (
                      bowl['Bowling_Skill'] == 'Left-arm medium-fast') | (
                      bowl['Bowling_Skill'] == 'Left-arm fast') | (bowl['Bowling_Skill'] == 'Left-arm medium')]
    bd['Pacer'] = 1
    bowl = bowl[(bowl['Bowling_Skill'] == 'chinaman') | (bowl['Bowling_Skill'] == 'Right-arm offbreak') | (
            bowl['Bowling_Skill'] == 'Left-arm offbreak') | (bowl['Bowling_Skill'] == 'Legbreak googly') | (
                        bowl['Bowling_Skill'] == 'Legbreak') | (
                        bowl['Bowling_Skill'] == 'Slow left-arm orthodox') | (
                        bowl['Bowling_Skill'] == 'Slow left-arm chinaman')]
    # (pa[(pa['Toss_Decision']=="bat") & (pa['City_Name']=='Bangalore')][['Match_Winner_Id','Toss_Winner_Id']]
    bowl['Spinner'] = 1
    # print bowl
    bowl = bowl.append(bd)
    # print bowl
    bo_problem = pulp.LpProblem('IPLAuction', pulp.LpMaximize)
    bowl = bowl.fillna(0)
    # print bowl
    decision_variables = []
    y = 0;
    for rownumber, row in bowl.iterrows():
        v = str('x' + str(y))
        v = pulp.LpVariable(str(v), lowBound=0, upBound=1, cat='Integer')
        decision_variables.append(v)
        y = y + 1

    print decision_variables
    # maximization function
    player_rating = ""
    for rownum, row in bowl.iterrows():
        for i, j in enumerate(decision_variables):
            if rownum == i:
                f = row['Bowler_Index'] * j
                player_rating += f
    # print decision_variables

    bo_problem += player_rating
    # print bo_problem
    # print ("Optimization function: " + str(player_rating))
    # # bowl['Bowler_price']=bowl['Bowler_price']/100
    # # print bowl
    # # bowl=bowl.fillna(9999)

    budget = int(bowl_budget)  # input("Enter Budget")
    cur_cost = ""
    for rownum, row in bowl.iterrows():
        for i, j in enumerate(decision_variables):
            if rownum == i:
                # print row['Base_Price']
                f = row['Bowler_price'] * j
                # print f
                cur_cost += f
    # print cur_cost
    bo_problem += (cur_cost <= budget)

    no_bowls = int(spin_no) # input("Enter Number of Spinners")
    cur_bowl = ""
    for rownum, row in bowl.iterrows():
        for i, j in enumerate(decision_variables):
            if rownum == i:
                # print row['Base_Price']
                f = row['Spinner'] * j
                # print f
                cur_bowl += f
    # print cur_cost
    bo_problem += (no_bowls == cur_bowl)

    no_pacers = int(nobl)  # input("Enter Number of Pacers")
    cur_pacers = ""
    for rownum, row in bowl.iterrows():
        for i, j in enumerate(decision_variables):
            if rownum == i:
                # print row['Base_Price']
                f = row['Pacer'] * j
                # print f
                cur_pacers += f
    # print cur_cost
    bo_problem += (no_pacers == cur_pacers)

    # # no_spinner=input("Enter the number of Spinners")
    # # cur_spinner=""
    # # for rownum, row in d.iterrows():
    # # 	for i, j in enumerate(decision_variables):
    # # 		if rownum == i:
    # # 			# print row['Base_Price']
    # # 			f = row['Spinner']*j
    # # 			# print f
    # # 			cur_spinner += f
    # # # print cur_cost
    # # bo_problem += (no_spinner == cur_spinner)
    result = bo_problem.solve()

    # # d=bowl[(bowl['Bowling_Skill']=='chinaman') || (bowl['Bowling_Skill']=='Right-arm offbreak')|| (bowl['Bowling_Skill']=='Left-arm offbreak') || (bowl['Bowling_Skill']=='Legbreak googly' ) || (bowl['Bowling_Skill']=='Legbreak') ]

    # # print bo_problem

    # result = bo_problem.solve()

    assert result == pulp.LpStatusOptimal
    # print("Status:", LpStatus[bo_problem.status])
    # print("Optimal Solution to the problem: ", value(bo_problem.objective))
    # print ("Individual decision_variables: ")
    # for v in bo_problem.variables():
    # print(v.name, "=", v.varValue)

    variable_name = []
    variable_value = []

    for v in bo_problem.variables():
        variable_name.append(v.name)
        variable_value.append(v.varValue)

    df = pd.DataFrame({'variable': variable_name, 'value': variable_value})
    for rownum, row in df.iterrows():
        value = re.findall(r'(\d+)', row['variable'])
        df.loc[rownum, 'variable'] = int(value[0])

    # df = df.sort_index(by='variable')

    # append results
    for rownum, row in bowl.iterrows():
        for results_rownum, results_row in df.iterrows():
            if rownum == results_row['variable']:
                bowl.loc[rownum, 'decision'] = results_row['value']

    bfound = bowl[bowl['decision'] == 1][['Player_Name', 'Bowler_Index', 'Bowler_price', 'Spinner', 'Pacer']]
    bfound.to_csv("bowler_selected.csv")

    # For Bowlers
    # For Bowlers
    # For Bowlers
    '''bowl = pd.read_csv('optibowl.csv')
    bowl['Bowler_price'].replace(' ', '0', inplace=True)
    bowl.Base_Price = pd.to_numeric(bowl['Bowler_price'], errors='coerse')
    # bowl=pd.read_csv('optibowl.csv')
    # print bat.dtypes
    bowl['Bowler'] = 1

    bo_problem = pulp.LpProblem('IPLAuction', pulp.LpMaximize)

    decision_variables = []
    for rownumber, row in bowl.iterrows():
        v = str('x' + str(rownumber))
        v = pulp.LpVariable(str(v), lowBound=0, upBound=1, cat='Integer')
        decision_variables.append(v)

    # maximization function
    player_rating = ""
    for rownum, row in bowl.iterrows():
        for i, j in enumerate(decision_variables):
            if rownum == i:
                f = row['Bowler_Index'] * j
                player_rating += f
    # print decision_variables
    bo_problem += player_rating
    # print bo_problem
    # print ("Optimization function: " + str(player_rating))
    bowl['Bowler_price'] = bowl['Bowler_price'] / 100
    # print bowl
    bowl = bowl.fillna(9999)

    budget = int(bowl_budget)
    cur_cost = ""
    for rownum, row in bowl.iterrows():
        for i, j in enumerate(decision_variables):
            if rownum == i:
                # print row['Base_Price']
                f = row['Bowler_price'] * j
                # print f
                cur_cost += f
    # print cur_cost
    bo_problem += (cur_cost <= budget)

    no_bowls = int(nobl)
    cur_bowl = ""
    for rownum, row in bowl.iterrows():
        for i, j in enumerate(decision_variables):
            if rownum == i:
                # print row['Base_Price']
                f = row['Bowler'] * j
                # print f
                cur_bowl += f
    # print cur_cost
    bo_problem += (no_bowls == cur_bowl)

    # print bo_problem

    result = bo_problem.solve()

    assert result == pulp.LpStatusOptimal
    # print("Status:", LpStatus[bo_problem.status])
    # print("Optimal Solution to the problem: ", value(bo_problem.objective))
    # print ("Individual decision_variables: ")
    # for v in bo_problem.variables():
    # print(v.name, "=", v.varValue)

    variable_name = []
    variable_value = []

    for v in bo_problem.variables():
        variable_name.append(v.name)
        variable_value.append(v.varValue)

    df = pd.DataFrame({'variable': variable_name, 'value': variable_value})
    for rownum, row in df.iterrows():
        value = re.findall(r'(\d+)', row['variable'])
        df.loc[rownum, 'variable'] = int(value[0])

    # df = df.sort_index(by='variable')

    # append results
    for rownum, row in bowl.iterrows():
        for results_rownum, results_row in df.iterrows():
            if rownum == results_row['variable']:
                bowl.loc[rownum, 'decision'] = results_row['value']

    dfbowl= (bowl[bowl['decision'] == 1][['Player_Name', 'Bowler_Index', 'Country']])
    dfbowl=dfbowl[['Player_Name', 'Bowler_Index', 'Country']]
    dfbowl.to_csv("bowler_selected.csv")
'''








