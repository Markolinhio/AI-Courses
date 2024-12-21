import random

RED = [3,3,3,3,3,6]
BLUE = [2,2,2,5,5,5]
OLIVE = [1,4,4,4,4,4]
DICE_NAMES = ["Red", "Blue", "Olive"]

# The function initialises the random number generator
# with the seed number input by the user.

def init_die():
    siemenluku = int(input("Give a seed for the dice.\n"))
    random.seed(siemenluku)

# IMPLEMENT THE MISSING FUNCTIONS HERE
def roll(die):
    number = random.randint(0,5)
    if die == 0:    
        return RED[number]
    elif die == 1:
        return BLUE[number]
    else:
        return OLIVE[number]
def simulate_singles(die1, die2, rolls):
    win1 = 0
    win2 = 0
    draw = 0
    for i in range (0, rolls):
        a = roll(die1)
        b = roll(die2)
        if (a > b):
            win1 += 1
        elif (a < b):
            win2 += 1
        else:
            draw += 1
    return win1, win2, draw
def simulate_doubles(die1, die2, rolls):
    win1 = 0
    win2 = 0
    draw = 0
    for i in range (0, rolls):
        a = roll(die1)
        b = roll(die1)
        c = roll(die2)
        d = roll(die2)
        if ((a+b) > (c+d)):
            win1 += 1
        elif ((a+b) < (c+d)):
            win2 += 1
        else:
            draw += 1
    return win1, win2, draw
# Calls for the appropriate simulation function
# and prints the outcome of the simulation.

def simulate_and_print_result(die1, die2, rolls, simulation_function, header):
    wins1, wins2, draws = simulation_function(die1, die2, rolls)
    print(header)
    print("Player 1 used {:s} die and won {:d} times, so {:.1f}% of the rolls.".format(DICE_NAMES[die1],wins1,wins1/rolls*100))
    print("Player 2 used {:s} die and won {:d} times, so {:.1f}% of the rolls.".format(DICE_NAMES[die2],wins2,wins2/rolls*100))
    if draws != 0:
        print("{:d} draws, so {:.2f}% of the rolls.".format(draws, draws/rolls*100))

def main():
    print("Welcome to a non-transitive dice simulation.")
    init_die()
    print("The dice:")
    print("{:d} for {:s}: {:}".format(0 ,DICE_NAMES[0], RED))
    print("{:d} for {:s}: {:}".format(1 ,DICE_NAMES[1], BLUE))
    print("{:d} for {:s}: {:}".format(2 ,DICE_NAMES[2], OLIVE))

    choice1 = int(input("Choose a die for player 1:\n"))
    choice2 = int(input("Choose a die for player 2:\n"))
    rolls  = int(input("How many rolls to simulate?\n"))
    simulate_and_print_result(choice1, choice2, rolls, simulate_singles, "Singles:")
    simulate_and_print_result(choice1, choice2, rolls, simulate_doubles, "Doubles:")

main()