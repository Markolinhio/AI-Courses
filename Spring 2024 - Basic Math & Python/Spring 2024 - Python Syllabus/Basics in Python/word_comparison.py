def compare_words(word1, word2):
    same_letters_same_place = 0
    same_letters_different_place = 0
    """ Write your code here """
    if (len(word1) != len(word2)):
        return -1, -1
    else:
        word1 = word1.lower()
        word2 = word2.lower()
        index = []
        count = 0
        for i in range (0, len(word1)):
            if word1[i] == word2[i]:
                index.append(i)
        same_letters_same_place = len(index)
        for i in index:
                word1 = word1[0:i-count] + word1[i+1-count:len(word1)]
                word2 = word2[0:i-count] + word2[i+1-count:len(word2)]
                count += 1
        #print(word1)
        #print(word2)
        for i in range (0, len(word1)):
            char1 = word1[i]
            #print("Char 1:",char1)
            found = False
            for j in range (0, len(word2)):
                char2 = word2[j]
                #print("Char 2:", char2)
                if char1 == char2 and not found:
                    if i == 0:
                        word1 = word1[0:0] + " " + word1[1:len(word1)]
                        word2 = word2[0:j] + " " + word2[j+1:len(word2)]
                    elif j == 0:
                        word1 = word1[0:i] + " " + word1[i+1:len(word1)]
                        word2 = word2[0:0] + " " + word2[1:len(word2)]
                    else:
                        word1 = word1[0:i] + " " + word1[i+1:len(word1)]
                        word2 = word2[0:j] + " " + word2[j+1:len(word2)]
                    found = True
                    same_letters_different_place += 1
        return same_letters_same_place, same_letters_different_place

def main():
    print("Let's compare some words!")
    cont = True
    while cont:
        first_word = input("First word:\n")
        second_word = input("Second word:\n")
        matching_letters_same_place, matching_letters_diff_place = compare_words(first_word, second_word)
        print("compare_words('{:s}', '{:s}') returned the values {:d}, {:d}".format(first_word, second_word, matching_letters_same_place, matching_letters_diff_place))
        print("Enter to continue, Q+Enter to quit")
        if input().upper().startswith('Q'):
            cont = False


main()