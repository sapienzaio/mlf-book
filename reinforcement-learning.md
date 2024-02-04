---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Reinforcement learning

A common assumption in classical machine learning problems is that
the data-generating distribution is fixed and stationary.
Under this assumption, one estimates the parameters of the model on a *train* dataset and
evaluates the expected performance of the model on a held-out *test* dataset.
In finance, however, it is common knowledge that the data-generating distribution of the prices is always changing —
a model trained to predict the stock market using data from the year 1999 is not expected to have a good performance
when deploying it in the year 2020.
One way to overcome the limitation of this classical point of view is to build agents that *adapt* to their environment.
In this chapter, an *adaptive* agent is one that, having committed to a decision,
obtain a reward (or punishment) based on this decision, and update its behaviour based on the reward.
In this chapter, we study agents whose goal is to maximise their expected future returns.
This mode of behaviour is called sequential decision making and a 
the class of algorithms to tackle them is called  reinforcement learning
{cite}`sutton2018reinforcement`.

+++

## The RL components

```{tikz}
\tikzstyle{reward}=[shape=circle,draw=blue!50,fill=blue!10]
\tikzstyle{action}=[shape=circle,draw=green,fill=green!10]
\tikzstyle{state}=[shape=circle,draw=red!50,fill=red!10]
\tikzstyle{gru}=[shape=rectangle,draw=black!50,fill=lime!10]
\tikzstyle{obs}=[shape=circle,draw=blue!50,fill=blue!10]
\tikzstyle{lightedge}=[<-,dotted]
\tikzstyle{mainstate}=[state,thick]
\tikzstyle{mainedge}=[<-,thick]

\node[reward] (r1) at (0,3) {$~r_{t}~$};%(3,2.5)

\node[] (sinit) at (-4,0) {$\dots$};
\node[state,scale=1] (s0) at (-2,0) {$S_{t-1}$};
\node[state,scale=1] (s1) at (0,0) {$~~S_{t}~$};
\node[state,scale=1] (s2) at (2,0) {$S_{t+1}$}; %(3,0)
\node[] (s3) at (4,0) {$\dots$};

\node[action] (a1) at (0,1.5) {$~a_{t}~$};%(1.5,1.25) 

\draw [->] (sinit) to (s0);
\draw [->] (s0) to (s1);
\draw [->] (s1) to (s2);
\draw [->] (s2) to (s3);

\draw [->] (s1) to (a1);
\draw [->] (a1) to (r1);
\draw [->] (a1) to  (s2);

\draw [->] (s2) to (r1);
\draw [->] (s1) to [out=165, in=165] (r1.west);
\draw [->] (a1) to (r1);
```

+++

## Stationary $\epsilon$-greedy bandit

The classical $\epsilon$-greedy algorithm makes a decision based on the rule

$$
    \hat a_{t+1} =
    \begin{cases}
        \arg\max_a q_t(a) & \text{w.p. } 1 - \epsilon \\
        a \sim \text{Mult}({\cal A}) & \text{w.p. } \epsilon
    \end{cases}
$$


with $0 < \epsilon \ll 1$ and
$q_t(a) = \mathbb{E}[R_t \vert {\cal A}_t = a]$
the average return obtained after
choosing arm $a$.

We estimate $q_t(a)$ recursively as

$$
\begin{aligned}
q_t(a)
&= \frac{1}{N_t(a)}\sum_{\tau=1}^t r_\tau \mathbb{1}(\hat a_\tau = a)\\
&= \frac{1}{N_t(a)}\left(\frac{N_t(a)-1}{N_t(a)-1}\sum_{\tau=1}^{t-1}r_\tau \mathbb{1}(\hat a_\tau = a) + r_t\right)\\
&= \frac{1}{N_t(a)}\left(\frac{N_t(a)-1}{N_{t-1}(a)}\sum_{\tau=1}^{t-1}r_\tau \mathbb{1}(\hat a_\tau = a) + r_t\right)\\
&= \frac{1}{N_t(a)}\Big((N_t(a)-1) q_{t-1}(a) + r_N\Big)\\
&= \frac{1}{N_t(a)}\left(N_t(a)\,q_{t-1}(a) - q_{t-1}(a) + r_N\right)\\
&= q_{t-1}(a) + \frac{1}{N_t(a)}\Big(r_t - q_{t-1}(a)\Big),
\end{aligned}
$$

with $N_t(a) = \sum_{\tau=1}^t \mathbb{1}(\hat a_{\tau} =a)$

+++

## Non-Stationary $\epsilon$-greedy bandit

The classical $\epsilon$-greedy algorithm makes a decision based on the rule

$$
    \hat a_{t+1} =
    \begin{cases}
        \arg\max_a q_t(a) & \text{w.p. } 1 - \epsilon \\
        a \sim \text{Mult}({\cal A}) & \text{w.p. } \epsilon
    \end{cases}
$$


with $0 < \epsilon \ll 1$ and
$q_t(a) = \mathbb{E}[R_t \vert {\cal A}_t = a]$
the average return obtained after
choosing arm $a$.

We estimate $q_t(a)$ recursively as
$$
\begin{aligned}
q_t(a)
&= q_{t-1}(a) + \alpha_t\Big(r_t - q_{t-1}(a)\Big),
\end{aligned}
$$

with $\alpha_t > 0$.

+++

## (Stationary) contextual $\epsilon$-greedy bandit

The tabular $\epsilon$-greedy algorithm makes a decision based on the rule

$$
    \hat a_{t+1} =
    \begin{cases}
        \arg\max_a q_t(s_t, a) & \text{w.p. } 1 - \epsilon \\
        a \sim \text{Mult}({\cal A}) & \text{w.p. } \epsilon,
    \end{cases}
$$
for a given context (state) $s_t$, and
$0 < \epsilon \ll 1$ and
$q_t(s, a) = \mathbb{E}[R_t \vert {\cal S} = s,{\cal A}_t = a]$
the average return obtained after
choosing arm $a$ and observing $s_t$.

We estimate $q_t(a)$ recursively as

$$
\begin{aligned}
q_t(s_t, a)
&= \frac{1}{N_t(s_t, a)}\sum_{\tau=1}^t r_\tau \mathbb{1}(\hat a_\tau = a)\mathbb{1}(s_\tau = s_t)\\
&= \frac{1}{N_t(s_t, a)}\left(\frac{N_t(s_t, a)-1}{N_t(s_t, a)-1}\sum_{\tau=1}^{t-1}r_\tau \mathbb{1}(\hat a_\tau = a)\mathbb{1}(s_\tau = s_t) + r_t\right)\\
&= \frac{1}{N_t(s_t, a)}\left(\frac{N_t(s_t, a)-1}{N_{t-1}(s_t, a)}\sum_{\tau=1}^{t-1}r_\tau \mathbb{1}(\hat a_\tau = a)\mathbb{1}(s_\tau = s_t) + r_t\right)\\
&= \frac{1}{N_t(s_t, a)}\Big((N_t(s_t, a)-1) q_{t-1}(s_t, a) + r_N\Big)\\
&= \frac{1}{N_t(s_t, a)}\left(N_t(s_t, a)\,q_{t-1}(s_t, a) - q_{t-1}(s,a) + r_N\right)\\
&= q_{t-1}(s_t, a) + \frac{1}{N_t(s_t, a)}\Big(r_t - q_{t-1}(s_t, a)\Big),
\end{aligned}
$$

with $N_t(s_t, a) = \sum_{\tau=1}^t \mathbb{1}(\hat a_{\tau} =a)\mathbb{1}(\hat s_{\tau} = s_t)$

```{bibliography}
```
