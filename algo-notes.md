# Adaptive Color Plane Interpolation by Hamilton and Adams

## High level

multi-step process with classifiers (edge detection?), first and second order derivatives.

Three step process:
 1. populate green channel (luminance)
 2. populate red or blue (chrominance)
 3. populate remaining chrominance channel

## First pass: calculate green luminance data for red and blue positions

We calculate the green value for the image position A5, where Ax is either red for all positions or blue.

```
      A1
      G2
A3 G5 A5 G6 A7
      G8
      A9
```

First: calculate the classifiers `alpha` and `beta`

```
alpha = abs(-A3 + 2*A5 - A7)
beta  = abs(-A1 + 2*A5 - A9)
```

Then, if alpha < beta:

```
G5 = (G4 + G6) / 2 + (-A3 + 2*A5 - A7) / 2
```

else if alpha > beta:

```
G5 = (G2 + G8) / 2 + (-A1 + 2*A5 - A9) / 2
```

else // alpha == beta

```
G5 = (G2 + G4 + G6 + G6) / 4 + (-A1 - A3 + 4*A5 - A7 - A9) / 8
```

Note that the signed versions of alpha and beta are reused in each of those calculations. The alpha == beta case is almost the average of the other cases.

## Second pass: calculate blue chromaticity channel

### Overview and cases

First we calculate the missing blue values. Here we use the following pattern, where C5 is red, Gx is green and Ax is blue.

```
A1 G2 A3
G4 C5 G6
A7 G8 A9
```

We have blue values for Ax and need to calculate it for the remaining 5 positions. There are three cases:
 * G2 and G8, where we have blue values to the right and left (A1, A3 and A7, A9)
 * G4 and G6, where we have blue values to the top and bottom (A1, A7 and A3, A9)
 * C5, where we have blue values on the diagonal (A1, A3, A7, A9)

### Case 1: Horizontal neighbours

We want to calculate the blue values for the positions labeled G2 and G8 in our pattern, which would be apropriately named A2 and A8. Here we have horizontal neighbours A1, A3 and A7, A9 of the same color. These are calculated as follows:

```
A2 = (A1 + A3) / 2 + (-G1 + 2*G2 - G3) / 2
A8 = (A7 + A9) / 2 + (-G7 + 2*G8 - G9) / 2
```

### Case 2: Vertical neighbours

We want to calculate the blue values for the positions labeled G4 and G6 in our pattern, which would be apropriately named A4 and A6. Here we have vertical neighbours A1, A7 and A3, A9 of the same color. These are calculated as follows:

```
A4 = (A1 + A7) / 2 + (-G1 + 2*G4 - G7) / 2
A6 = (A3 + A9) / 2 + (-G3 + 2*G6 - G9) / 2
```

### Case 3: Diagonals

To calculate the blue value for C5, we first calculate the classifiers `alpha` and `beta`:

```
alpha = abs(-G3 + 2*G5 - G7)
beta  = abs(-G1 + 2*G5 - G9))
```

Then, if alpha < beta:

```
C5 = (A3 + A7) / 2 + (-G3 + 2*G5 - G7) / 2
```

else if alpha > beta:

```
C5 = (A1 + A9) / 2 + (-G1 + 2*G5 - G9) / 2
```

else // alpha == beta

```
C5 = (A1 + A3 + A7 + A9) / 4 + (-G1 - G3 + 4*G5 - G7 - G9) / 8
```

Again, note the similarities arround alpha, beta and the alpha == beta case.

## Third pass: repeat the second pass for the remaining color

Done.

# Implementation

This is the Algo that describes what needs to be calculated. Algo wise, the green pass needs to happen first, it's inputs are needed for the blue and red pass. Those two only depend on the green pass I think and are independent so can maybe be calculated in the same loop, or in parallel.

It may be easier due to indexing to use two loops for the green pass - one for the even indizes, one for the odd, or in other words, one for the positions where red color is measured, one for where blue is measured. I'm not sure how I would do that in one loop, but maybe you'll figure something out.

For the red and blue pass, it may be good to do something similar, or do the horizontal pass wirst, then the vertical.

Or loop over all pixels and decide with a switch or similar which calculation to apply.

# General advice

It helps to first write down something like this, the algorythm in your own words, at a high level. Then think about how to actually code it, how to structure it, which loops you do in which order, again on paper. Then the actual coding is quick, because you know what to do. So go write that part of your thesis.

I think it helps here to extend the matrix class with the color channel, that maps even better to the naming convention the paper uses. Or change the convention to something that makes sense to you and helps you to understand what happens.
