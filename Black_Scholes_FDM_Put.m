clc, clear, close all
%% Given Data
K = 100; % Strike Price
Stock_Price = 150; % Current Stock Price
sigma = 60/100; % Implied Volatility
r = 0.02; % Assume 2 Percent. Close to US 10 Year Treasury Yield

T = 150/365.25; % Years Until Expiration
dtao = 1/365.25; % 1 Day Time Steps

L = 2*Stock_Price; % Maximum Stock Price
n_blocks = 500; % Number of Elements
dS = L/n_blocks; % Price Step

n_nodes = n_blocks + 1; % Mesh Centered Grid
n_free = n_nodes - 2; % Number of Free Nodes

S = (0:dS:L)'; % Stock Price Vector from $0 to Maximum Stock Price
tao = (0:dtao:T)'; % Tao Vector, where tao = T - t
n_time = length(tao); % Number of Time Steps

V_1 = K*exp(-r*tao); % Option Price at Stock Price = $0 - Boundary Condition ($)
V_n = 0; % Option Price at Maximum Stock Price - Boundary Condition ($)
V = max(K-S, 0); % Free Global Option Price Vector at tao = 0 - Initial Condition ($)

V_total = zeros(n_nodes, n_time); % Initialize Option Price Storage Matrix for All Time Steps
V_total(:, 1) = V; % Initial Condition at tao = 0
V_total(1, :) = V_1; % Boundary Condition at Manually Set Maximum Stock Price
V = V(2:end-1); % Remove the fixed DOFs from V for FDM

%% Matrix Initializations
A = zeros(n_free, n_free); % Free Global 'Stiffness' Matrix
B = zeros(n_free, 1); % Forcing Vector

for t = 2:n_time
    %% Global Matrix Assembly
    theta = 1/2; % Central difference method
    for i = 1:n_free
        %% Options Price Constants (Theta Differencing)
        a1 = theta*S(i+1)/(2*dS)*(r-sigma^2*S(i+1)/dS);         % Constant for V(i-1, t+dt)
        a2 = 1/dtao + theta*((sigma*S(i+1)/dS)^2+r);            % Constant for V(i, t+dt)
        a3 = -theta*S(i+1)/(2*dS)*(r+sigma^2*S(i+1)/dS);        % Constant for V(i+1, t+dt)
        b1 = (1-theta)*S(i+1)/(2*dS)*(-r+sigma^2*S(i+1)/dS);    % Constant for V(i-1, t)
        b2 = 1/dtao - (1-theta)*((sigma*S(i+1)/dS)^2+r);        % Constant for V(i, t)
        b3 = (1-theta)*S(i+1)/(2*dS)*(r+sigma^2*S(i+1)/dS);     % Constant for V(i+1, t)

        %% Free DOF A Matrix Construction
        if i == n_free
            A(i, i-1) = a1;
            A(i, i) = a2;
        elseif i == 1
            A(i, i) = a2;
            A(i, i+1) = a3;
        else
            A(i, i-1) = a1;
            A(i, i) = a2;
            A(i, i+1) = a3;
        end

        %% Free DOF B Matrix Construction
        if i == n_free
            B(i) = b1*V(i-1) + b2*V(i) + (b3-a3)*V_n;
        elseif i == 1
            B(i) = b1*V_1(t-1) + b2*V(i) + b3*V(i+1) - a1*V_1(t);
        else
            B(i) = b1*V(i-1) + b2*V(i) + b3*V(i+1);
        end
    end

    V = A\B;
    V_total(2:end-1, t) = V;
end
V_total;
%% Post Processing
% Filter V_total to Only Include More Realistic Stock Price Movements
% Range from -20% to +20% Current Stock Price
lower = Stock_Price*0.8;
upper = Stock_Price*1.2;
lowerbound_index = find(S == max(S(S <= lower)));
upperbound_index = find(S == min(S(S >= upper)));
V_filtered = V_total(lowerbound_index:upperbound_index, end:-1:1);
S_filtered = S(lowerbound_index:upperbound_index);

% Find the FDM Approximated Option Price at Current Stock Price and tao = T
current_index = find(S == max(S(S <= Stock_Price)));
% Linear Interpolation to Find Option Price In-Between Nodes
Option_Price = (V_total(current_index+1,end)-V_total(current_index,end))...
    /dS*(Stock_Price-S(current_index)) + V_total(current_index,end);
V_gainloss = (V_filtered - Option_Price)/Option_Price*100; % Percent Gain/Loss

%% Plotting
figure(1)
tao_days = tao*365.25; % Change time scale to days for graphing
surf(tao_days, S, V_total)
xlabel('Time Until Expiration (Days)')
ylabel('Stock Price ($)')
zlabel('Option Price ($)')
title('Black-Scholes Model - Put Option Pricing')

%% Print Results
fprintf('The current option price is $%.2f', Option_Price)