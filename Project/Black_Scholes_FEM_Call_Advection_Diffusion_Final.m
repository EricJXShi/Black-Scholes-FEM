clc, clear, close all
%% Given Data
K = 1000; % Strike Price
Stock_Price = 1145.45; % Current Stock Price
sigma = 62.51/100; % Implied Volatility
r = 0.02441; % Assume 2 Percent. Close to US 10 Year Treasury Yield

T = 255/365.25; % Years Until Expiration
dtao = 25.5/365.25; % 1 Day Time Steps

L = 2*Stock_Price; % Maximum Stock Price
n_blocks = 10; % Number of Elements
dS = L/n_blocks; % Price Step

n_nodes = n_blocks + 1; % Linear Approximation
fixed_dofs = [1, n_nodes]; % Node numbers of fixed DOFs
free_dofs = setxor(1:n_nodes,fixed_dofs); % Node numbers of free DOFs

S = (0:dS:L)'; % Stock Price Vector from $0 to Maximum Stock Price
tao = (0:dtao:T)'; % Tao Vector, where tao = T - t
n_time = length(tao); % Number of Time Nodes

V_1 = 0; % Option Price at Stock Price = $0 - Boundary Condition ($)
V_n = L - K*exp(-r*tao); % Option Price at Maximum Stock Price - Boundary Condition ($)
V = max(S-K, 0); % Free Global Option Price Vector at tao = 0 - Initial Condition ($)

V_total = zeros(n_nodes, n_time); % Initialize Option Price Storage Matrix for All Time Steps
V_total(:, 1) = V; % Initial Condition at tao = 0
V_total(end, :) = V_n; % Boundary Condition at Manually Set Maximum Stock Price

syms SP
dplus_T = (log(SP/K)+(r+sigma^2/2)*(T))/(sigma*(T)^(1/2));
dminus_T = (log(SP/K)+(r-sigma^2/2)*(T))/(sigma*(T)^(1/2));
Exact_Solution = SP*normcdf(dplus_T) - K*exp(-r*T)*normcdf(dminus_T);

%% Define Connectivity of Global Free DOFs
conn = zeros(n_blocks, 2); % Initialize element to node connectivity
for i = 1:n_blocks
    conn(i, 1) = i; % Local node 1
    conn(i, 2) = i+1; % Local node 2
end

%% Define Shape Functions
syms n 
N1n = 1/2*(1-n); N2n = 1/2*(1+n); % Linear shape functions in local coordinates
dndS = 2/dS; % Relationship between dS and dn in the form of dn/dS
dN1dn = diff(N1n); dN2dn = diff(N2n); % Derivative of shape functions wrt n
dN1dS = dN1dn*dndS; dN2dS = dN2dn*dndS; % Derivative of shape functions wrt S

%% Derive Parts of 'Stiffness' Matrix using Method of Weighted Residuals
k11eB1 = int((1-n)*dN1dS*N1n, [-1 1]); k12eB1 = int((1-n)*dN1dS*N2n, [-1 1]);
k21eB1 = int((1-n)*dN2dS*N1n, [-1 1]); k22eB1 = int((1-n)*dN2dS*N2n, [-1 1]);
kB1 = [k11eB1, k12eB1; k21eB1, k22eB1];

k11eB2 = int((1+n)*dN1dS*N1n, [-1 1]); k12eB2 = int((1+n)*dN1dS*N2n, [-1 1]);
k21eB2 = int((1+n)*dN2dS*N1n, [-1 1]); k22eB2 = int((1+n)*dN2dS*N2n, [-1 1]);
kB2 = [k11eB2, k12eB2; k21eB2, k22eB2];

k11eD = int(N1n^2, [-1 1]); k12eD = int(N1n*N2n, [-1 1]);
k21eD = int(N2n*N1n, [-1 1]); k22eD = int(N2n^2, [-1 1]);
kD = [k11eD, k12eD; k21eD, k22eD];

k11eE1 = int((1-n)^2*dN1dS^2, [-1 1]); k12eE1 = int((1-n)^2*dN1dS*dN2dS, [-1 1]);
k21eE1 = int((1-n)^2*dN2dS*dN1dS, [-1 1]); k22eE1 = int((1-n)^2*dN2dS^2, [-1 1]);
kE1 = [k11eE1, k12eE1; k21eE1, k22eE1];

k11eE2 = int((1-n)*(1+n)*dN1dS^2, [-1 1]); k12eE2 = int((1-n)*(1+n)*dN1dS*dN2dS, [-1 1]);
k21eE2 = int((1-n)*(1+n)*dN2dS*dN1dS, [-1 1]); k22eE2 = int((1-n)*(1+n)*dN2dS^2, [-1 1]);
kE2 = [k11eE2, k12eE2; k21eE2, k22eE2];

k11eE3 = int((1+n)^2*dN1dS^2, [-1 1]); k12eE3 = int((1+n)^2*dN1dS*dN2dS, [-1 1]);
k21eE3 = int((1+n)^2*dN2dS*dN1dS, [-1 1]); k22eE3 = int((1+n)^2*dN2dS^2, [-1 1]);
kE3 = [k11eE3, k12eE3; k21eE3, k22eE3];

alpha_coeff = sigma^2/dS*[2, -1; -1, 2];

%% Global 'Stiffness' Matrix Assembly (Apply Theta Differencing)
A = zeros(n_nodes, n_nodes); % Initialize Global 'Stiffness' Matrix for the next time step
B = zeros(n_nodes, n_nodes); % Initialize Global 'Stiffness' Matrix for the current time step
theta = 1/2; % Central difference method
for i = 1:n_blocks
    S1 = S(i); S2 = S(i+1); % Local node stock price coordinates
    enodes = conn(i, :); % Element to node connectivity for local to global mapping
    beta = kD\((sigma^2-r)*(1/2*S1*kB1 + 1/2*S2*kB2) + 1/8*...
        sigma^2*(S1^2*kE1 + 2*S1*S2*kE2 + S2^2*kE3) + r*kD); % [beta] in the hand calculation
    ket1 = alpha_coeff\(1/dtao - beta*(1-theta)); % Local 'Stiffness' Matrix for the current time step
    ket2 = alpha_coeff\(1/dtao + beta*theta); % Local 'Stiffness' Matrix for the next time step
    A(enodes, enodes) = A(enodes, enodes) + ket2; % Assembly of the global 'stiffness' matrix
    B(enodes, enodes) = B(enodes, enodes) + ket1; % Assembly of the global forcing vector
end

%% Global 'Stiffness' Matrix Partitioning
A_E = A(fixed_dofs, fixed_dofs);
A_EF = A(fixed_dofs, free_dofs);
A_FE = A(free_dofs, fixed_dofs);
A_F = A(free_dofs, free_dofs);

%% Solving for Option Pricing for all Stock Price Increments and Time Steps
for t = 2:n_time
    V_E2 = V_total(fixed_dofs, t);
    RS = B*V;
    V_F2 = A_F\(RS(2:end-1) - A_FE*V_E2); % Free DOF Option pricing at next time step
    V_total(2:end-1, t) = V_F2;
    V = V_total(:, t);
end
V_total

%% Post Processing
% Find the FEM Approximated Option Price at Current Stock Price and tao = T
current_index = find(S == max(S(S <= Stock_Price)));
% Linear Interpolation to Find Option Price In-Between Nodes
Option_Price = (V_total(current_index+1,end)-V_total(current_index,end))...
    /dS*(Stock_Price-S(current_index)) + V_total(current_index,end);

%% Plotting
figure(1)
tao_days = tao*365.25; % Change time scale to days for graphing
surf(tao_days, S, V_total)
xlabel('Time Until Expiration (Days)')
ylabel('Stock Price ($)')
zlabel('Option Price ($)')
title('Black-Scholes Model - Call Option Pricing')

figure(2)
fplot(SP, Exact_Solution, [0, L], 'LineWidth', 2)
hold on
plot(S, V_total(:, end), 'r--', 'LineWidth', 2)
hold off
xlabel('Stock Price ($)')
ylabel('Option Price ($)')
title('Black-Scholes Equation - Call Option Pricing at tao = T')
legend('Black-Scholes Equation', 'FE Approximation')
ylim([0 inf])

%% Print Results
fprintf('The current option price using Finite Element Method is $%.4f\n', Option_Price)
fprintf('The current option price using Black-Scholes Equation is $%.4f\n', subs(Exact_Solution, Stock_Price))