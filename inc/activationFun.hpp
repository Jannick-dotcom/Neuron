#pragma once

#include <string>
#include <cmath>

class ActivationFunction {
public:
    virtual double operator()(double x) = 0;
    virtual double derivative(double x) = 0;
    std::string name;
};

class Linear : public ActivationFunction {
public:
    double operator()(double x) override {
        return x;
    }
    double derivative(double x) override {
        return 1.0;
    }
    Linear() {
        name = "Linear";
    }
};

class Sigmoid : public ActivationFunction {
public:
    double operator()(double x) override {
        return 1.0 / (1.0 + exp(-x));
    }
    double derivative(double x) override {
        return (*this)(x) * (1.0 - (*this)(x));
    }
    Sigmoid() {
        name = "Sigmoid";
    }
};

class Tanh : public ActivationFunction {
public:
    double operator()(double x) override {
        return tanh(x);
    }
    double derivative(double x) override {
        return 1.0 / pow(cosh(x),2);
    }
    Tanh() {
        name = "Tanh";
    }
};

class ReLU : public ActivationFunction {
public:
    double operator()(double x) override {
        return x > 0.0 ? x : 0.0;
    }
    double derivative(double x) override {
        return x > 0.0 ? 1.0 : 0.0;
    }
    ReLU() {
        name = "ReLU";
    }
};

class LeakyReLU : public ActivationFunction {
public:
    double operator()(double x) override {
        return x > 0.0 ? x : 0.01 * x;
    }
    double derivative(double x) override {
        return x > 0.0 ? 1.0 : 0.01;
    }
    LeakyReLU() {
        name = "LeakyReLU";
    }
};