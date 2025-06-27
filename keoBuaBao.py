import random 

choices = {'rock': 0, 'paper': 1, 'scissors': 2}

botTotal = 0
userTotal = 0

rounds = eval(input("how many rounds do you want to play versus bot? "))


def KeoBuaBao(userInput):
    global botTotal, userTotal
    botChoice = random.choice(list(choices))
    botNum= choices[botChoice]
    userChoice = choices[userInput]
    # rock paper scissor algo ref: https://learningpenguin.net/2020/02/06/a-simple-algorithm-for-calculating-the-result-of-rock-paper-scissors-game/ 
    if (botNum + 1) % 3 == userChoice:
        print(f"bot chose {botChoice}.User won!")
        userTotal = userTotal + 1 
    elif botNum == userChoice:
        print(f"Bot also chose {botChoice}. It is a draw")
    else:
        print(f"Bot chose {botChoice}. Bot won!")
        botTotal = botTotal + 1


for i in range(rounds):
    print(f"round {i+1}:")
    userInput = input("please choose either rock, paper or scissors: ")
    while userInput not in choices:
        userInput = input("invalid choices, please choose either rock, paper or scissors: ").lower()
    KeoBuaBao(userInput)
    print("------------")

winner = " " 
if userTotal > botTotal:
    winner = "User"
elif userTotal < botTotal:
    winner = "Bot"
else:
    winner = "No one wins"


print(
        f"""
        Bot total score:{botTotal} 
        User total score: {userTotal}
        Winner: {winner}
        """
        )



