import random

def guessing_game():
    secret_number = random.randint(1, 30)  # Secret number between 1 and 30
    tries = 0
    max_tries = 15

    print("Welcome to the Guessing Game!")
    print("I have selected a number between 1 and 30. You have 15 tries to guess it!")

    while tries < max_tries:
        try:
            guess = int(input(f"Attempt {tries + 1}: Enter your guess: "))
        except ValueError:
            print("Invalid input! Please enter a valid number.")
            continue

        if guess < 1 or guess > 30:
            print("Please guess a number between 1 and 30.")
            continue

        tries += 1

        if guess < secret_number:
            print("Your guess is too small!")
        elif guess > secret_number:
            print("Your guess is too large!")
        else:
            print(f"Congratulations! You guessed the number in {tries} tries.")
            break
    else:
        print(f"Sorry! You've used all {max_tries} attempts. The secret number was {secret_number}.")

guessing_game()
