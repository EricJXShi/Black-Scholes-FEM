clc, clear, close all
%% Given Data
K = 25; % Strike Price
Stock_Price = 39.19; % Current Stock Price
sigma = 127.23/100; % Implied Volatility
r = 0.02; % Assume 2 Percent. Close to US 10 Year Treasury Yield

T = 20/365.25; % Years Until Expiration
dtao = 1/365.25; % 1 Day Time Steps

tao = (0:dtao:T)'; % Tao Vector, where tao = T - t
n_time = length(tao); % Number of Time Nodes
n_time_steps = n_time - 1; % Number of Time Steps

n_blocks = 30; % Number of Elements
n_nodes = n_blocks + 1; % Linear Approximation
fixed_dofs = [1, n_nodes]; % Node numbers of fixed DOFs
free_dofs = setxor(1:n_nodes,fixed_dofs); % Node numbers of free DOFs

L = 2*Stock_Price; % Maximum Stock Price
dS = (L-0.0000000001)/n_blocks; % Price Step
S = (0.0000000001:dS:L)'; % Stock Price Vector from $0 to Maximum Stock Price
x = zeros(n_nodes, n_time);
for i = 1:n_time
    x(:, i) = log(S(:)) + (r-1/2*sigma^2)*tao(i); % Space Vector, where x is a function of stock price and tao
end

dx = zeros(n_blocks, 1);
for i = 1:n_blocks
    dx(i) = x(i+1, 1) - x(i, 1);
end
    
F_1 = 0; % Option Price at $0 stock price - Boundary Condition
F_n = (L - K*exp(-r*tao)).*exp(r*tao); % Option Price at max stock price - Boundary Condition
F = max(exp(max(x(:, 1), 0))-K, 0); % Option Price at tao = 0 - Initial Condition
%F = max(exp((1/2 + r/sigma^2)*x(:, 1)) - exp((-1/2 + r/sigma^2)*x(:, 1)), 0);

F_total = zeros(n_nodes, n_time); % Initialize Option Price Storage Matrix for All Time Steps
F_total(:, 1) = F; % Initial Condition at tao = 0
F_total(end, :) = F_n; % Boundary Condition at Manually Set Maximum Stock Price

dplus_T = (log(Stock_Price/K)+(r+sigma^2/2)*(T))/(sigma*(T)^(1/2));
dminus_T = (log(Stock_Price/K)+(r-sigma^2/2)*(T))/(sigma*(T)^(1/2));
Exact_Solution = Stock_Price*normcdf(dplus_T) - K*exp(-r*T)*normcdf(dminus_T);

%% Define Connectivity of Global Free DOFs
conn = zeros(n_blocks, 2); % Initialize element to node connectivity
for i = 1:n_blocks
    conn(i, 1) = i; % Local node 1
    conn(i, 2) = i+1; % Local node 2
end

%% Define Shape Functions
syms n 
N1n = 1/2*(1-n); N2n = 1/2*(1+n); % Linear shape functions in local coordinates
dndx = 2./dx; % Relationship between dx and dn in the form of dn/dx
dN1dn = diff(N1n); dN2dn = diff(N2n); % Derivative of shape functions wrt n
dN1dx = dN1dn*dndx; dN2dx = dN2dn*dndx; % Derivative of shape functions wrt x

%% Derive Parts of 'Stiffness' Matrix using Method of Weighted Residuals
k11eA = int(N1n^2, [-1 1]); k12eA = int(N1n*N2n, [-1 1]);
k21eA = int(N2n*N1n, [-1 1]); k22eA = int(N2n^2, [-1 1]);
kA = [k11eA, k12eA; k21eA, k22eA];

k11eC = double(int(dN1dx.^2, [-1 1])); k12eC = double(int(dN1dx.*dN2dx, [-1 1]));
k21eC = double(int(dN2dx.*dN1dx, [-1 1])); k22eC = double(int(dN2dx.^2, [-1 1]));

alpha_coeff = sigma^2*[2, -1; -1, 2]; % Coefficient to rearrange the alpha term in the derivation so that it cancels out

%% Global 'Stiffness' Matrix Assembly (Apply Theta Differencing)
A = zeros(n_nodes, n_nodes); % Initialize Global 'Stiffness' Matrix for the next time step
B = zeros(n_nodes, n_nodes); % Initialize Global 'Stiffness' Matrix for the current time step
theta = 1/2; % Central difference method
for i = 1:n_blocks
    kC = [k11eC(i), k12eC(i); k21eC(i), k22eC(i)];
    enodes = conn(i, :); % Element to node connectivity for local to global mapping
    beta = kA\(1/2*sigma^2*kC); % [beta] in the hand calculation
    ket1 = alpha_coeff\(1/dtao - (1-theta)*beta)*dx(i); % Local 'Stiffness' Matrix for the current time step
    ket2 = alpha_coeff\(1/dtao + theta*beta)*dx(i); % Local 'Stiffness' Matrix for the next time step
    A(enodes, enodes) = A(enodes, enodes) + ket2; % Assembly of the global 'stiffness' matrix for the next time step
    B(enodes, enodes) = B(enodes, enodes) + ket1; % Assembly of the global 'stiffness' matrix for the current time step
end

%% Global 'Stiffness' Matrix Partitioning
A_E = A(fixed_dofs, fixed_dofs);
A_EF = A(fixed_dofs, free_dofs);
A_FE = A(free_dofs, fixed_dofs);
A_F = A(free_dofs, free_dofs);

B_E = B(fixed_dofs, fixed_dofs);
B_EF = B(fixed_dofs, free_dofs);
B_FE = B(free_dofs, fixed_dofs);
B_F = B(free_dofs, free_dofs);

for j = 1:n_time
    %% Solving for Option Pricing for all Stock Price Increments and Time Steps
    if j >= 2
        F_E = F(fixed_dofs); % Boundary Conditions at current time step
        F_F = F(free_dofs); % Free DOF Option pricing at current time step
        F_E2 = F_total(fixed_dofs, j); % Boundary Conditions at next time step
        F_F2 = A_F\(B_FE*F_E + B_F*F_F - A_FE*F_E2); % Free DOF Option pricing at next time step
        F_total(2:end-1, j) = F_F2;
        F = F_total(:, j);
    end
end
F_total;

%% Post Processing
% Filter V_total to Only Include More Realistic Stock Price Movements
% Range from -20% to +20% Current Stock Price
V_total = zeros(n_nodes, n_time);
for i = 1:n_time
    V_total(:, i) = F_total(:, i)*exp(-r*tao(i));
end
V_total

lower = Stock_Price*0.8;
upper = Stock_Price*1.2;
lowerbound_index = find(S == max(S(S <= lower)));
upperbound_index = find(S == min(S(S >= upper)));
V_filtered = V_total(lowerbound_index:upperbound_index, end:-1:1);
S_filtered = S(lowerbound_index:upperbound_index, :);

% Find the FDM Approximated Option Price at Current Stock Price and tao = T
current_index = find(S == max(S(S <= Stock_Price)));
% Linear Interpolation to Find Option Price In-Between Nodes
dS_current = S(current_index+1)-S(current_index);
Option_Price = (V_total(current_index+1,end)-V_total(current_index,end))...
    /dS_current*(Stock_Price-S(current_index)) + V_total(current_index,end);
V_gainloss = (V_filtered - Option_Price)/Option_Price*100; % Percent Gain/Loss

%% Plotting
figure(1)
tao_days = tao*365.25; % Change time scale to days for graphing
surf(tao_days, S, V_total)
xlabel('Time Until Expiration (Days)')
ylabel('Stock Price ($)')
zlabel('Option Price ($)')
title('Black-Scholes Model - Call Option Pricing')

%% Print Results
fprintf('The current option price is $%.2f', Option_Price)