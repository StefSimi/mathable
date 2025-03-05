import cv2 as cv
import numpy as np
import os
import sys

currentTestFile=""
currentTestString=""
currentTask1String=""

trainFolder="testare/"

globali=0
counter=0

globalTrue=0
globalTotal=0

player1Score=0
player2Score=0
scoreMatrix=[[-1 for _ in range(14)] for _ in range(14)]
scoreMatrix[6][6]=1
scoreMatrix[6][7]=2
scoreMatrix[7][6]=3
scoreMatrix[7][7]=4

currentScore=0

modifierMatrix=[[0 for _ in range(14)] for _ in range(14)]
#3X
#Corners
modifierMatrix[0][0]=3
modifierMatrix[0][13]=3
modifierMatrix[13][0]=3
modifierMatrix[13][13]=3
#Edge up
modifierMatrix[0][6]=3
modifierMatrix[0][7]=3
#Edge down
modifierMatrix[13][6]=3
modifierMatrix[13][7]=3
#Edge left
modifierMatrix[6][0]=3
modifierMatrix[7][0]=3
#Edge right
modifierMatrix[6][13]=3
modifierMatrix[7][13]=3

#2X
#Up Left
modifierMatrix[1][1]=2
modifierMatrix[2][2]=2
modifierMatrix[3][3]=2
modifierMatrix[4][4]=2

#Up Right
modifierMatrix[1][12]=2
modifierMatrix[2][11]=2
modifierMatrix[3][10]=2
modifierMatrix[4][9]=2

#Down Left
modifierMatrix[12][1]=2
modifierMatrix[11][2]=2
modifierMatrix[10][3]=2
modifierMatrix[9][4]=2

#Down Right
modifierMatrix[12][12]=2
modifierMatrix[11][11]=2
modifierMatrix[10][10]=2
modifierMatrix[9][9]=2

# + 4
# - 5
# * 6
# / 7

# Up signs
modifierMatrix[3][6]=4
modifierMatrix[4][7]=4
modifierMatrix[3][7]=6
modifierMatrix[4][6]=6
modifierMatrix[2][5]=5
modifierMatrix[2][8]=5
modifierMatrix[1][4]=7
modifierMatrix[1][9]=7

#Down signs
modifierMatrix[9][6]=4
modifierMatrix[10][7]=4
modifierMatrix[9][7]=6
modifierMatrix[10][6]=6
modifierMatrix[11][5]=5
modifierMatrix[11][8]=5
modifierMatrix[12][4]=7
modifierMatrix[12][9]=7

#Left signs
modifierMatrix[6][4]=4
modifierMatrix[7][3]=4
modifierMatrix[6][3]=6
modifierMatrix[7][4]=6
modifierMatrix[5][2]=5
modifierMatrix[8][2]=5
modifierMatrix[4][1]=7
modifierMatrix[9][1]=7

#Right signs
modifierMatrix[6][10]=4
modifierMatrix[7][9]=4
modifierMatrix[6][9]=6
modifierMatrix[7][10]=6
modifierMatrix[5][11]=5
modifierMatrix[8][11]=5
modifierMatrix[4][12]=7
modifierMatrix[9][12]=7


#for row in modifierMatrix:
#        print(row)
def areNeighborsValid(n1i,n1j,n2i,n2j):
    if n1i<0 or n1i >13 or n1j<0 or n1j>13 or n2i<0 or n2i >13 or n2j<0 or n2j>13 :
        return False
    else:
        if scoreMatrix[n2i][n2j]==-1 or scoreMatrix[n1i][n1j] ==-1:
            return False
        else:
            return True

def getNewScore(i,j):
    score=0

    modifier=modifierMatrix[i][j]
    #print(str(modifier)+"Mod")
    neighbor1i = 0
    neighbor1j = 0
    neighbor2i = 0
    neighbor2j = 0

    #left
    edgeCaseProtection = 0
    neighbor1i=i
    neighbor1j=j-1
    neighbor2i=i
    neighbor2j=j-2
    if areNeighborsValid(neighbor1i,neighbor1j,neighbor2i,neighbor2j):
        if scoreMatrix[neighbor2i][neighbor2j]+scoreMatrix[neighbor1i][neighbor1j]==scoreMatrix[i][j]:
            if modifier != 5 and modifier !=6 and modifier !=7 and edgeCaseProtection==0:
                score+=scoreMatrix[i][j]
                edgeCaseProtection=1
        if scoreMatrix[neighbor2i][neighbor2j]*scoreMatrix[neighbor1i][neighbor1j]==scoreMatrix[i][j]:
            if modifier != 4 and modifier !=5 and modifier !=7 and edgeCaseProtection==0:
                score+=scoreMatrix[i][j]
                edgeCaseProtection=1
        if abs(scoreMatrix[neighbor2i][neighbor2j]-scoreMatrix[neighbor1i][neighbor1j])==scoreMatrix[i][j]:
            if modifier != 4 and modifier !=6 and modifier !=7 and edgeCaseProtection==0:
                score+=scoreMatrix[i][j]
                edgeCaseProtection=1
        if scoreMatrix[neighbor2i][neighbor2j]!=0 and scoreMatrix[neighbor1i][neighbor1j] and scoreMatrix[i][j]!=0:
            #print(scoreMatrix[neighbor2i][neighbor2j] / scoreMatrix[neighbor1i][neighbor1j])
            #print(scoreMatrix[neighbor1i][neighbor1j] / scoreMatrix[neighbor2i][neighbor2j])
            if float(scoreMatrix[neighbor2i][neighbor2j]/scoreMatrix[neighbor1i][neighbor1j])==float(scoreMatrix[i][j]) or float(scoreMatrix[neighbor1i][neighbor1j]/scoreMatrix[neighbor2i][neighbor2j])==float(scoreMatrix[i][j]):
                if modifier != 4 and modifier !=5 and modifier !=6 and edgeCaseProtection==0:
                    score+=scoreMatrix[i][j]
                    edgeCaseProtection=1

    # right
    edgeCaseProtection = 0
    neighbor1i = i
    neighbor1j = j + 1
    neighbor2i = i
    neighbor2j = j + 2
    if areNeighborsValid(neighbor1i, neighbor1j, neighbor2i, neighbor2j):
        if scoreMatrix[neighbor2i][neighbor2j] + scoreMatrix[neighbor1i][neighbor1j] == scoreMatrix[i][j]:
            if modifier != 5 and modifier != 6 and modifier != 7 and edgeCaseProtection == 0:
                score += scoreMatrix[i][j]
                edgeCaseProtection = 1
        if scoreMatrix[neighbor2i][neighbor2j] * scoreMatrix[neighbor1i][neighbor1j] == scoreMatrix[i][j]:
            if modifier != 4 and modifier != 5 and modifier != 7 and edgeCaseProtection == 0:
                score += scoreMatrix[i][j]
                edgeCaseProtection = 1
        if abs(scoreMatrix[neighbor2i][neighbor2j] - scoreMatrix[neighbor1i][neighbor1j]) == scoreMatrix[i][j]:
            if modifier != 4 and modifier != 6 and modifier != 7 and edgeCaseProtection == 0:
                score += scoreMatrix[i][j]
                edgeCaseProtection = 1
        if scoreMatrix[neighbor2i][neighbor2j] != 0 and scoreMatrix[neighbor1i][neighbor1j] and scoreMatrix[i][j] != 0:
            #print(scoreMatrix[neighbor2i][neighbor2j] / scoreMatrix[neighbor1i][neighbor1j])
            #print(scoreMatrix[neighbor1i][neighbor1j] / scoreMatrix[neighbor2i][neighbor2j])
            if float(scoreMatrix[neighbor2i][neighbor2j]/scoreMatrix[neighbor1i][neighbor1j])==float(scoreMatrix[i][j]) or float(scoreMatrix[neighbor1i][neighbor1j]/scoreMatrix[neighbor2i][neighbor2j])==float(scoreMatrix[i][j]):
                if modifier != 4 and modifier != 5 and modifier != 6 and edgeCaseProtection == 0:
                    score += scoreMatrix[i][j]
                    edgeCaseProtection = 1

                    # up
    edgeCaseProtection = 0
    neighbor1i = i-1
    neighbor1j = j
    neighbor2i = i-2
    neighbor2j = j
    if areNeighborsValid(neighbor1i, neighbor1j, neighbor2i, neighbor2j):
        if scoreMatrix[neighbor2i][neighbor2j] + scoreMatrix[neighbor1i][neighbor1j] == scoreMatrix[i][j]:
            if modifier != 5 and modifier != 6 and modifier != 7 and edgeCaseProtection == 0:
                score += scoreMatrix[i][j]
                edgeCaseProtection = 1
        if scoreMatrix[neighbor2i][neighbor2j] * scoreMatrix[neighbor1i][neighbor1j] == scoreMatrix[i][j]:
            if modifier != 4 and modifier != 5 and modifier != 7 and edgeCaseProtection == 0:
                score += scoreMatrix[i][j]
                edgeCaseProtection = 1
        if abs(scoreMatrix[neighbor2i][neighbor2j] - scoreMatrix[neighbor1i][neighbor1j]) == scoreMatrix[i][j]:
            if modifier != 4 and modifier != 6 and modifier != 7 and edgeCaseProtection == 0:
                score += scoreMatrix[i][j]
                edgeCaseProtection = 1
        if scoreMatrix[neighbor2i][neighbor2j] != 0 and scoreMatrix[neighbor1i][neighbor1j] and scoreMatrix[i][j] != 0:
            #print(scoreMatrix[neighbor2i][neighbor2j] / scoreMatrix[neighbor1i][neighbor1j])
            #print(scoreMatrix[neighbor1i][neighbor1j] / scoreMatrix[neighbor2i][neighbor2j])
            if float(scoreMatrix[neighbor2i][neighbor2j]/scoreMatrix[neighbor1i][neighbor1j])==float(scoreMatrix[i][j]) or float(scoreMatrix[neighbor1i][neighbor1j]/scoreMatrix[neighbor2i][neighbor2j])==float(scoreMatrix[i][j]):
                if modifier != 4 and modifier != 5 and modifier != 6 and edgeCaseProtection == 0:
                    score += scoreMatrix[i][j]
                    edgeCaseProtection = 1

                    # down
    edgeCaseProtection = 0
    neighbor1i = i + 1
    neighbor1j = j
    neighbor2i = i + 2
    neighbor2j = j
    if areNeighborsValid(neighbor1i, neighbor1j, neighbor2i, neighbor2j):
        if scoreMatrix[neighbor2i][neighbor2j] + scoreMatrix[neighbor1i][neighbor1j] == scoreMatrix[i][j]:
            if modifier != 5 and modifier != 6 and modifier != 7 and edgeCaseProtection == 0:
                score += scoreMatrix[i][j]
                edgeCaseProtection = 1
        if scoreMatrix[neighbor2i][neighbor2j] * scoreMatrix[neighbor1i][neighbor1j] == scoreMatrix[i][j]:
            if modifier != 4 and modifier != 5 and modifier != 7 and edgeCaseProtection == 0:
                score += scoreMatrix[i][j]
                edgeCaseProtection = 1
        if abs(scoreMatrix[neighbor2i][neighbor2j] - scoreMatrix[neighbor1i][neighbor1j]) == scoreMatrix[i][j]:
            if modifier != 4 and modifier != 6 and modifier != 7 and edgeCaseProtection == 0:
                score += scoreMatrix[i][j]
                edgeCaseProtection = 1
        if scoreMatrix[neighbor2i][neighbor2j] != 0 and scoreMatrix[neighbor1i][neighbor1j] and scoreMatrix[i][j] != 0:
            #print(scoreMatrix[neighbor2i][neighbor2j] / scoreMatrix[neighbor1i][neighbor1j])
            #print(scoreMatrix[neighbor1i][neighbor1j] / scoreMatrix[neighbor2i][neighbor2j])
            if float(scoreMatrix[neighbor2i][neighbor2j]/scoreMatrix[neighbor1i][neighbor1j])==float(scoreMatrix[i][j]) or float(scoreMatrix[neighbor1i][neighbor1j]/scoreMatrix[neighbor2i][neighbor2j])==float(scoreMatrix[i][j]):
                if modifier != 4 and modifier != 5 and modifier != 6 and edgeCaseProtection == 0:
                    score += scoreMatrix[i][j]
                    edgeCaseProtection = 1

    if modifier==2:
        score*=2
    if modifier==3:
        score*=3

    return score

def show_image(title,image):
    image=cv.resize(image,(0,0),fx=1,fy=1)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def show_all_images():
    files=os.listdir(trainFolder)
    for file in files:
        if file[-3:]=='jpg':
            img = cv.imread(trainFolder+'/'+file)
            show_image('img',img)

def getPositionOutput(i,j):
    global counter
    letter=chr(ord('A') + j - 1)
    #print(str(i)+letter,end=" ")#+" "+str(counter))
    return (str(i)+letter)

def check_file_contents(currentTestFile, currentTestString):
    try:
        with open(trainFolder+'/'+currentTestFile, 'r') as file:
            file_contents = file.read()
        if(file_contents==currentTestString):
            global globalTrue
            globalTrue+=1
            return True
        else:
            print("True "+file_contents+"\nPredicted "+currentTestString+"\nFile"+currentTestFile)
            return False
    except FileNotFoundError:
        print(f"Error: File '{trainFolder+'/'+currentTestFile}' not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def create_document(fileName, fileString):
    try:
        with open(fileName, 'w') as file:
            file.write(fileString)
        print(f"File '{fileName}' created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                             flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=255)
    return rotated

def new_identify_image(target_image):
    total=93*93
    #print(target_image.shape)
    target_image = cv.threshold(target_image, 128, 255, cv.THRESH_BINARY)[1]

    best_match = None
    best_score = -1

    inner_size = 98
    margin = (100 - inner_size) // 2

    target_inner = target_image[margin:-margin, margin:-margin]
    #show_image('a', target_inner)
    #print(target_inner.shape)
    rotation_angles = [-5, -2, 0, 2, 5]
    for template_file in os.listdir('modified_templates'):
        template_path = os.path.join('modified_templates', template_file)
        template_image = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
        template_image = cv.threshold(template_image, 128, 255, cv.THRESH_BINARY)[1]
        template_inner = template_image[margin:-margin, margin:-margin]

        max_score = -1
        for angle in rotation_angles:
            rotated_template=rotate_image(template_inner,angle)
            for dx in range(-15, 16):  #      -10 11
                for dy in range(-18, 19):
                    shifted_template = np.roll(rotated_template, shift=(dy, dx), axis=(0, 1))
                    score = np.sum(target_inner == shifted_template)

                    if score > max_score:
                        max_score = score
                        ''' if float(max_score/total) >0.98:
                            #print("Yep Skip")
                            best_match = template_file[0:2]
                            if (best_match[0] == '0'):
                                best_match = best_match[1]
                            #print(str(best_match) + " new")
                            return best_match'''

        if max_score > best_score:
            best_match = template_file[0:2]
            if (best_match[0] == '0'):
                best_match = best_match[1]

            best_score = max_score
            #best_match = template_file
    #print(best_score/(93*93))
    #print(str(best_match)+" new")
    #0 3 60 7
    return best_match





def extrage_careu(image):
    copy=image.copy()
    frame_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    l = np.array([0, 0, 0])
    u = np.array([60, 255, 255])
    mask_table_hsv = cv.inRange(frame_hsv, l, u)
    #cv.imshow("Mask", cv.resize(mask_table_hsv, (0, 0), fx=0.2, fy=0.2))



    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image_m_blur = cv.medianBlur(mask_table_hsv, 3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5)
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)

    #mask_table_hsv=cv.cvtColor(mask_table_hsv, cv.COLOR_BGR2GRAY)

    _, thresh = cv.threshold(image_sharpened, 30, 255, cv.THRESH_BINARY)

    kernel = np.ones((4, 4), np.uint8)
    thresh = cv.erode(thresh, kernel)




    edges = cv.Canny(thresh, 200, 400)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0


    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    #stupid test
    #7 10->127 127
    #906 6->787 127
    #11 903->125 788
    #907 906 -> 787 792


    topDiff=top_right[0]-top_left[0]
    botDiff=bottom_right[0]-bottom_left[0]
    leftDiff=bottom_left[1]-top_left[1] #Jg diff
    rightDiff=bottom_right[1]-top_right[1] #Mid diff

    #Should probably be cos(angle)- sin(angle) or something
    #I guess if it's rotated then we reshape it to a regular square and then do the same thing but I cba it's 4 AM


    # I have brain damage
    top_left[0]+= 13.348/100*topDiff
    top_left[1]+= 13.102/100*leftDiff
    top_right[0] -= 13.237/100*topDiff
    top_right[1] += 13.444/100*rightDiff
    bottom_left[0] += 12.723/100*botDiff
    bottom_left[1] -= 12.878/100*botDiff
    bottom_right[0] -= 13.393/100*botDiff
    bottom_right[1] -= 12.667/100*rightDiff

    

    #Nvm not stupid it's actually better than before


    width = 1400
    height = 1400

    image_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    result = cv.warpPerspective(copy, M, (width, height))
    #result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

    return result


lines_horizontal=[]
for i in range(0,1401,100):
    l=[]
    l.append((0,i))
    l.append((1396,i))
    lines_horizontal.append(l)


lines_vertical=[]
for i in range(0,1401,100):
    l=[]
    l.append((i,0))
    l.append((i,1396))
    lines_vertical.append(l)

filled_board = np.zeros((14,14),dtype='uint8')

def determina_configuratie_careu_ox(thresh,lines_horizontal,lines_vertical):
    matrix = np.empty((14,14), dtype='str')
    frame_hsv = cv.cvtColor(thresh, cv.COLOR_BGR2HSV)
    thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    l = np.array([0, 0, 147])
    u = np.array([94, 85, 255])
    mask_table_hsv = cv.inRange(frame_hsv, l, u)
    imax=-1
    jmax=-1
    medieMax=-1
    yMinFinal=-1
    yMaxFinal=-1
    xMinFinal=-1
    xMaxFinal=-1
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 20
            y_max = lines_vertical[j + 1][1][0] - 20
            x_min = lines_horizontal[i][0][1] + 20
            x_max = lines_horizontal[i + 1][1][1] - 20
            printable = thresh[x_min-15:x_max+20, y_min-15:y_max+20].copy()
            #print(printable.shape)#53
            patch = mask_table_hsv[x_min:x_max, y_min:y_max].copy()
            Medie_patch=np.mean(patch)
            if Medie_patch>medieMax and filled_board[i][j]==0:

                medieMax=Medie_patch
                imax=i
                jmax=j
                yMinFinal=y_min
                xMinFinal=x_min
                yMaxFinal=y_max
                xMaxFinal=x_max

    filled_board[imax][jmax] = 1
    cv.rectangle(result, (yMinFinal - 20, xMinFinal - 20), (yMaxFinal + 20, xMaxFinal + 20), color=(255, 0, 0), thickness=5)
    global counter
    counter = counter + 1

    global currentTestString
    printable = thresh[xMinFinal - 15:xMaxFinal + 20, yMinFinal - 15:yMaxFinal + 20].copy()

    newNumber=new_identify_image(printable)
    scoreMatrix[imax][jmax]=int(newNumber)
    '''for row in scoreMatrix:
        print(row)'''
    moveScore=getNewScore(imax,jmax)
    global currentScore
    currentScore+=moveScore
    #print(str(moveScore)+" move "+str(currentScore)+" score")

    outputText=getPositionOutput(imax+1,jmax+1)+" "+newNumber
    global currentTask1String
    currentTask1String=getPositionOutput(imax+1,jmax+1)
    currentTestString=outputText

    target_image = cv.threshold(printable, 128, 255, cv.THRESH_BINARY)[1]
    #show_image('a',target_image)

    return matrix

def vizualizare_configuratie(result,lines_horizontal,lines_vertical):
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            if filled_board[i][j] == 1:
                cv.rectangle(result, (y_min, x_min), (y_max, x_max), color=(255, 0, 0), thickness=5)


testfileName=1
scoreFile=(trainFolder+'/'+str(testfileName)+'_turns.txt')
scoreContents=[]
try:
    with open(scoreFile, 'r') as file:
        for line in file:
            name, turn = line.strip().split()
            scoreContents.append({"name": name, "turn": int(turn)})
except FileNotFoundError:
    print(f"Error: File '{trainFolder + '/' + currentTestFile}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")

scoreContents.append({"name": "debug", "turn": 337})
scoreContentsIndex=1



#print(scoreContents[0]["name"])  # Player1
#print(scoreContents[0]["turn"])  # 1


files=os.listdir(trainFolder)
for file in files:
    if file[-3:]=='jpg':
        img = cv.imread(trainFolder+'/'+file)
        result=extrage_careu(img)
        determina_configuratie_careu_ox(result,lines_horizontal,lines_vertical)
        vizualizare_configuratie(result,lines_horizontal,lines_vertical)
        for line in  lines_vertical :
            cv.line(result, line[0], line[1], (0, 255, 0), 5)
        for line in  lines_horizontal :
            cv.line(result, line[0], line[1], (0, 0, 255), 5)
        globali = globali + 1
        if(scoreContents[scoreContentsIndex]["turn"]==globali+1):
            try:
                with open("results/" + str(testfileName) + '_scores.txt', 'a') as file2:
                    file2.write(scoreContents[scoreContentsIndex-1]["name"]+" "+str(scoreContents[scoreContentsIndex-1]["turn"])+" "+str(currentScore)+"\n")
            except FileNotFoundError:
                print(f"Error: File '{trainFolder + '/' + currentTestFile}' not found.")
            except Exception as e:
                print(f"An error occurred: {e}")


            print(scoreContents[scoreContentsIndex-1]["name"]+" "+str(scoreContents[scoreContentsIndex-1]["turn"])+" "+str(currentScore))
            scoreContentsIndex+=1
            currentScore=0
            #print("reset")



        currentTestFile=file[:-3]
        currentTestFile+="txt"
        #check_file_contents(currentTestFile,currentTestString)
        #print(currentTask1String)
        print(currentTestString)
        #create_document("results/"+currentTestFile,currentTask1String)
        create_document("results/"+currentTestFile,currentTestString)
        globalTotal+=1
        #print(str(globalTrue)+"/"+str(globalTotal))
        #print("")
        if(globali==50):
            try:
                with open("results/" + str(testfileName) + '_scores.txt', 'a') as file2:
                    file2.write(scoreContents[scoreContentsIndex - 1]["name"] + " " + str(scoreContents[scoreContentsIndex - 1]["turn"]) + " " + str(currentScore))
            except FileNotFoundError:
                print(f"Error: File '{testfileName}' not found.")
            except Exception as e:
                print(f"An error occurred: {e}")
            print(scoreContents[scoreContentsIndex - 1]["name"] + " " + str(scoreContents[scoreContentsIndex - 1]["turn"]) + " " + str(currentScore))
            currentScore=0
            testfileName+=1
            scoreFile = (trainFolder+'/' + str(testfileName) + '_turns.txt')
            scoreContents = []
            try:
                with open(scoreFile, 'r') as file:
                    for line in file:
                        name, turn = line.strip().split()
                        scoreContents.append({"name": name, "turn": int(turn)})
            except FileNotFoundError:
                print(f"Error: File '{scoreFile}' not found.")
            except Exception as e:
                print(f"An error occurred: {e}")
            scoreContents.append({"name": "debug", "turn": 337})
            scoreContentsIndex = 1
            print("")




            globali=0
            filled_board = np.zeros((14, 14), dtype='uint8')
            player1Score = 0
            player2Score = 0
            scoreMatrix = [[-1 for _ in range(14)] for _ in range(14)]
            scoreMatrix[6][6] = 1
            scoreMatrix[6][7] = 2
            scoreMatrix[7][6] = 3
            scoreMatrix[7][7] = 4

