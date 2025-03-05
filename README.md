# Mathable Score Calculator

## About

A computer vision project that automates the process of calculating the score of a given Mathable game, by detecting newly placed pieces using different image manipulation techniques, such as image sharpening, changing HSV values, and perspective warping

## What is Mathable

  

<img src="https://github.com/user-attachments/assets/045194e5-855c-432d-94bc-5f486cd6aa0b" width="50%" />


Mathable is a board game similar to Scrabble, the only major difference being that instead of spelling out words, you instead have to create mathematical equations.<br>
The 4 center squares are always the same, acting as a starting point.<br>
Players will then take turns placing numbers on the board in order to complete one of four basic operations: addition, subtraction, multiplication or division. The result of the equation is added to the current player's total score.<br>
There are different types of special squares:<br>
-Score multipliers: The score of the equation whose solution is placed on top of it will be multiplied by the modifier's amount;
-Operation restrictors: A piece may only be placed on this square if it completes a specific type of equation.

## Computer vision
Board detection: <br>
-The board is identified using the HSV color space and contour detection; <br>
-The 4 largest contours are selected as board corners; <br>
-The playable area is extracted by shifting the coordonates inward; <br>
Piece detection: <br>
-The image is converted to HSV and a mask is applied to tedect added numbers;<br>
-Each cell is extracted as an individual patch;<br>
-The brightest patch will be considered the newly added piece;<br>
Number recognition:<br>
-A template-matching approach with pixel shifts is used;<br>
-Thresholding and slight rotations are used to improve accuracy;<br>
-Processing time is reduced by using a 98% correlation threshold;<br>
Score calculation:<br>
-A check is being done to see if the last placed number forms a valid arithmetic operation;<br>
-The numbers are stored in a matrix, and the validity of the neighbors is verified;<br>
-Special cells are being handled to restrict possible operations or apply multipliers;<br>

All of these proceudres result in a 99.5% accuracy.


Different methods have been used to extract relevant information from the given images: 




