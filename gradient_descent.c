/*
 * gradient_descent.c
 *
 * A tiny machine learning program written from scratch in C.
 *
 * The model is one line:
 *
 *     prediction = x * w
 *
 * The program trains the weight w until the predictions match a small
 * dataset whose hidden rule is y = 2x.
 *
 * Training uses gradient descent: the slope of the cost is found with the
 * first-principles definition of a derivative, and w is moved a small step
 * downhill, again and again.
 *
 */

#include <stdio.h>

/* ---------------- Training data ----------------
 * Each row is {x, y}: an input and the correct output.
 * The hidden rule is y = 2x. The program is never told this.
 */
float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

/* Number of rows: total bytes of the array / bytes in one row */
int rows = sizeof(train) / sizeof(train[0]);

/* ---------------- Training settings ---------------- */
#define START_W   3.0f    /* first guess for the weight                 */
#define DW        0.001f  /* tiny nudge used to measure the slope       */
#define RATE      0.01f   /* learning rate: how big each step is        */
#define STEPS     51      /* how many training steps to run (0 to 50)   */

/* ---------------- Cost function ----------------
 * Measures how wrong the model is for a given weight w,
 * using mean squared error (MSE):
 *
 *     cost(w) = average of (y - x*w)^2 over all rows
 *
 * Squaring makes every error positive, so errors cannot cancel out,
 * and it gives the cost a single lowest point at the best w.
 */
float cost(float w)
{
    float sum_sq_err = 0;

    for (int i = 0; i < rows; i++) {
        float x = train[i][0];
        float y = train[i][1];
        float prediction = x * w;
        float err = y - prediction;
        sum_sq_err += err * err;
    }

    return sum_sq_err / rows;
}

int main(void)
{
    float w = START_W;

    printf("Training y = x * w on %d data points\n\n", rows);

    for (int step = 0; step < STEPS; step++) {

        /* Show progress every 10 steps */
        if (step % 10 == 0) {
            printf("Step %2d: w = %.4f, cost = %.6f\n", step, w, cost(w));
        }

        /* Slope of the cost at w, from the first-principles derivative:
         *     dcost/dw = (cost(w + dw) - cost(w)) / dw
         * A positive slope means a bigger w makes the cost worse.
         */
        float dcost = (cost(w + DW) - cost(w)) / DW;

        /* Gradient descent update: step w against the slope, downhill */
        w = w - RATE * dcost;
    }

    /* Use the trained model on an input it has never seen */
    float new_x = 10;
    printf("\nLearned weight: w = %.4f (the true rule is w = 2)\n", w);
    printf("Prediction for x = %.0f: y = %.4f\n", new_x, new_x * w);

    return 0;
}
