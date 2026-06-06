# antStoch-I

A generative modeling framework implementing **Stochastic Interpolants** for mapping between arbitrary probability distributions.

## Overview
This repository implements the mathematical framework of Stochastic Interpolants as defined by Albergo & Vanden-Eijnden (2023). It unifies normalizing flows (ODEs) and diffusion processes (SDEs) by defining interpolation paths of the form:

$$x_t = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) z$$

## Project Structure
* `.gitignore`: Project ignore configuration.
* `README.md`: Project documentation.
