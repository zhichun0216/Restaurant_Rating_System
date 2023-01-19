# Restaurant Rating System (Mandani System)

## Input

1. Sentiment (1 - 10) : Trapezoid
   Bad [-1, 0, 3, 5],
   Normal [3, 4, 6, 7],
   Good [5, 7, 10, 11],

2. Service (1 - 10) : Trapezoid
   Bad [-1, 0, 3, 5],
   Normal [3, 4, 6, 7],
   Good [5, 7, 10, 11],

3. Food Quality (1 - 10) : Triangular + Trapezoid + Triangular
   Bad [-1, 0, 5],
   Normal [3, 4, 6, 7],
   Good [5, 10, 11],

4. Price (5 - 60) : Gaussian MF
   Low [5, 10],
   Normal [32.5, 10],
   High [60, 10],

5. Environment (1 - 10) : Z-Shaped + Guassian + S-Shaped
   Bad [1, 4],
   Normal [5, 2],
   Good [7, 10],

## Output

Restaurent Rating (1 - 10) :
