#include <stdio.h>
#include <stdlib.h>

#define MAX_ROWS 500


int read_csv(double X[MAX_ROWS], double Y[MAX_ROWS]) {

    FILE *file = fopen("placement.csv", "r");

    if (file == NULL) {
        printf("No Such File Exists. \n");
        return;
    }

    int count = 0;
    char line[1024];

    if (fgets(line, sizeof(line), file) == NULL) {
        printf("Error reading header. \n");
        fclose(file);
        return;
    }

    while (fgets(line, sizeof(line), file)) {

        double temp_x;
        double temp_y;

        if (sscanf(line, "%lf,%lf", &temp_x, &temp_y) == 2) {

            X[count] = temp_x;
            Y[count] = temp_y;
            count++;

            if (count >= MAX_ROWS) {
                printf("Maximum Limit Reached. \n");
                break;
            }
        }
        else {
            printf("Error in parsing Lines. \n");
        }
    }

    fclose(file);

    return count;

}

typedef struct LinearRegression {

    double m;
    double b;

}LinearRegression;

void initialize(LinearRegression * LR) {

    LR->m = 0;
    LR->b = 0;
}

void train_test_split(int rows, int test_size, double X[rows], double Y[rows], double X_train[rows - test_size], double Y_train[rows - test_size], double X_test[test_size], double Y_test[test_size]) {

    for (int i = 0; i < (rows - test_size); i++) {

        X_train[i] = X[i];
        Y_train[i] = Y[i];
    }

    for (int i = (rows - test_size), j = 0;  i < rows; i++, j++) {

        X_test[j] = X[i];
        Y_test[j] = Y[i];
    }
}

double mean(int size, double arr[size]) {

    double sum = 0;

    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }

    double SIZE = (double)size;

    double mean = (sum / SIZE);
    return mean;
}

void fit(LinearRegression * LR, int size, double x_train[size], double y_train[size]) {

    double numerator = 0;
    double denominator = 0;

    double x_mean = mean(size, x_train);
    double y_mean = mean(size, y_train);

    for(int i = 0; i < size; i++) {
        numerator += ((x_train[i] - x_mean) * (y_train[i] - y_mean));
        denominator += pow((x_train[i] - x_mean), 2);
    }

    LR->m = numerator / denominator;
    LR->b = y_mean - (LR->m * x_mean);

}

double predict(double x_test, LinearRegression * LR) {

    return ((x_test * LR->m) + LR->b);

}

double MAE(int size, double Y_test[size], double Y_predict[size]) {

    double numerator = 0;

    for (int i = 0; i < size; i++) {
        numerator += fabs(Y_test[i] - Y_predict[i]);
    }

    double SIZE = (double)size;

    return (numerator / SIZE);
}

double MSE(int size, double Y_test[size], double Y_predict[size]) {

    double numerator = 0;

    for (int i = 0; i < size; i++) {
        numerator += ((Y_test[i] - Y_predict[i]) * (Y_test[i] - Y_predict[i]));
    }

    double SIZE = (double)size;
    return (numerator / SIZE);
}

double R2_score(int size, double Y_test[size], double Y_predict[size]) {

    double numerator = 0;
    double y_mean= mean(size, Y_test);

    for (int i = 0; i < size; i++) {
        numerator += ((Y_test[i] - y_mean) * (Y_test[i] - y_mean));
    }

    double SIZE = (double)size;
    double ss_mean = (numerator / SIZE);

    return (1 - (MSE(size, Y_test, Y_predict) / ss_mean));
}

double adjusted_r2_score(int size, double Y_test[size], double Y_predict[size]) {

    double SIZE = (double)size;

    return  1 - (((1 - R2_score(size, Y_test, Y_predict)) * (SIZE - 1)) / (SIZE - 1 - 1));
}

void get_y_predict(int size, double Y_predict[size], double X_test[size], LinearRegression * LR) {

    for (int i = 0; i < size; i++) {
        Y_predict[i] = predict(X_test[i], LR);
    }
}

void displayTrain(int size, double X[size], double Y[size]) {

    printf("Training Set: \n");
    for (int i = 0; i < size; i++) {
        printf("%.2lf , %.2lf \n", X[i], Y[i]);
    }
    printf("\n");

}

void displayTest(int size, double X[size], double Y[size]) {

    printf("Testing Set: \n");
    for (int i = 0; i < size; i++) {
        printf("%.2lf , %.2lf \n", X[i], Y[i]);
    }
    printf("\n");
}

int main() {

    LinearRegression LR;

    initialize(&LR);

    double X[MAX_ROWS];
    double Y[MAX_ROWS];

    int rows = read_csv(X, Y);

    int ratio;
    printf("Enter the percentage of the Test Size[10-30]: \n");
    scanf("%d",&ratio);

    int test_size = ((rows / 100) * ratio);

    double X_train[rows - test_size];
    double X_test[test_size];
    double Y_train[rows - test_size];
    double Y_test[test_size];

    train_test_split(rows, test_size, X, Y, X_train, Y_train, X_test, Y_test);

    displayTrain((rows - test_size), X_train, Y_train);
    displayTest(test_size, X_test, Y_test);

    fit(&LR, (rows - test_size), X_train, Y_train);

    double Y_predict[test_size];
    get_y_predict(test_size, Y_predict, X_test, &LR);

    printf("1. coefficient: %lf \n", LR.m);
    printf("2. intercept: %lf \n", LR.b);
    printf("3. mean_absolute_error: %lf \n", MAE(test_size, Y_test, Y_predict));
    printf("4. mean_square_error: %lf \n", MSE(test_size, Y_test, Y_predict));
    printf("5. root_mean_square_error: %lf \n", pow(MSE(test_size, Y_test, Y_predict), 0.5));
    printf("6. r2_score: %lf \n", R2_score(test_size, Y_test, Y_predict));
    printf("7. adjusted_r2_score: %lf \n", adjusted_r2_score(test_size, Y_test, Y_predict));

    return 0;
}
