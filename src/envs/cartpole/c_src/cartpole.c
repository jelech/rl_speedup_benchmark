#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

// Constants matching Python implementation
#define GRAVITY 9.8
#define MASSCART 1.0
#define MASSPOLE 0.1
#define TOTAL_MASS (MASSCART + MASSPOLE)
#define LENGTH 0.5
#define POLEMASS_LENGTH (MASSPOLE * LENGTH)
#define FORCE_MAG 10.0
#define TAU 0.02
#define THETA_THRESHOLD_RADIANS (12.0 * 2.0 * M_PI / 360.0)
#define X_THRESHOLD 2.4

typedef struct {
    double x;
    double x_dot;
    double theta;
    double theta_dot;
    int steps_beyond_terminated;
} CartPoleState;

// Simple random float generator between min and max
double random_double(double min, double max) {
    double scale = rand() / (double) RAND_MAX;
    return min + scale * (max - min);
}

void CartPole_reset(CartPoleState* s) {
    s->x = random_double(-0.05, 0.05);
    s->x_dot = random_double(-0.05, 0.05);
    s->theta = random_double(-0.05, 0.05);
    s->theta_dot = random_double(-0.05, 0.05);
    s->steps_beyond_terminated = -1; // -1 indicates None
}

// Returns 1.0 for reward if not terminated, else 0.0 (handling the logic same as py)
// Returns is_terminated via pointer
// Returns state array via pointer
void CartPole_step(CartPoleState* s, int action, double* out_state, double* out_reward, bool* out_terminated) {
    double force = (action == 1) ? FORCE_MAG : -FORCE_MAG;
    double costheta = cos(s->theta);
    double sintheta = sin(s->theta);

    double temp = (force + POLEMASS_LENGTH * s->theta_dot * s->theta_dot * sintheta) / TOTAL_MASS;
    double thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (4.0/3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS));
    double xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

    // Euler integration
    s->x += TAU * s->x_dot;
    s->x_dot += TAU * xacc;
    s->theta += TAU * s->theta_dot;
    s->theta_dot += TAU * thetaacc;

    *out_terminated = (bool)(
        s->x < -X_THRESHOLD || 
        s->x > X_THRESHOLD || 
        s->theta < -THETA_THRESHOLD_RADIANS || 
        s->theta > THETA_THRESHOLD_RADIANS
    );

    if (!(*out_terminated)) {
        *out_reward = 1.0;
    } else if (s->steps_beyond_terminated == -1) {
        // Just fell
        s->steps_beyond_terminated = 0;
        *out_reward = 1.0;
    } else {
        s->steps_beyond_terminated += 1;
        *out_reward = 0.0;
    }

    // Update output state
    out_state[0] = s->x;
    out_state[1] = s->x_dot;
    out_state[2] = s->theta;
    out_state[3] = s->theta_dot;
}

CartPoleState* CartPole_new() {
    CartPoleState* s = (CartPoleState*)malloc(sizeof(CartPoleState));
    srand(time(NULL)); // Seed random
    return s;
}

void CartPole_free(CartPoleState* s) {
    free(s);
}
