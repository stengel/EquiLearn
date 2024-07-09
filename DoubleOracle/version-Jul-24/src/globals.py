import numpy as np


def initialize():
    global TOTAL_DEMAND, LOW_COST, HIGH_COST, TOTAL_STAGES,  GAMMA, NUM_ACTIONS, REWARDS_DIVISION_CONST
    global NUM_STOCHASTIC_ITER, NUM_MATRIX_ITER, N_EPISODES_BASE, N_EPISODES_LOAD, EPISODE_INCREASE_COEF
    # global  REPLAY_BUFFER_SIZE, PROB_BREAK_LIMIT_LN, CONVERGE_BREAK, PRINT_STEP,NUM_ADV_HISTORY, LR,ACTION_STEP
    # global BATCH_UPDATE_SIZE, BUFFER_PLAY_COEFFICIENT, NUM_PROCESS, NUM_MODEL_SAVE
    global CON_ACTIONS_RANGE, MODELS_DIR, LOG_DIR, NUM_TRACE_EQUILIBRIA, GAMES_DIR, NUM_PROCESS, MEMORY,DB_ITER_LIMIT
    global SPE_A, SPE_a, SPE_B, SPE_b, SPE_K, SPE_k, SPE_z, SPE_Y, ALG_PARAMS

    MEMORY=12
    TOTAL_DEMAND = 400
    LOW_COST = 57
    HIGH_COST = 71
    TOTAL_STAGES = 25
    GAMMA = 1
    NUM_ACTIONS = 20
    CON_ACTIONS_RANGE = 60
    # ACTION_STEP = 3
    
    REWARDS_DIVISION_CONST = 1000

    # N_EPISODES_BASE = 50
    # N_EPISODES_LOAD = 10
    # NUM_STOCHASTIC_ITER = 10
    # NUM_MATRIX_ITER = 2
    DB_ITER_LIMIT=20
    NUM_STOCHASTIC_ITER = 100
    NUM_MATRIX_ITER = 100
    N_EPISODES_BASE = 400_000
    N_EPISODES_LOAD = 200_000
    EPISODE_INCREASE_COEF = 1.2

    # NUM_EPISODES_RESET = NUM_EPISODES

    NUM_TRACE_EQUILIBRIA = 2
    # #HOW OFTEN THE MODEL SHOULD BE SAVED.
    # NUM_MODEL_SAVE=3

    NUM_PROCESS = 6

    MODELS_DIR = "models"
    LOG_DIR = "logs"
    GAMES_DIR = "games"

    # SPE coefficients

    SPE_a = [np.nan]*25
    SPE_A = [np.nan]*25
    SPE_b = [np.nan]*25
    SPE_B = [np.nan]*25
    SPE_k = [np.nan]*25
    SPE_K = [np.nan]*25
    SPE_z = [np.nan]*25
    SPE_Y = [np.nan]*25

    SPE_a[24] = 0.5
    SPE_A[24] = 0.25
    SPE_b[24] = 132
    SPE_B[24] = 68
    SPE_k[24] = 0.5
    SPE_K[24] = -0.5
    SPE_Y[24] = 0.25 * GAMMA

    for t in range(23, -1, -1):
        SPE_a[t] = (1-SPE_Y[t+1])/(2-SPE_Y[t+1])
        SPE_z[t] = GAMMA*(0.75-0.5*SPE_a[t])
        SPE_k[t] = (1-0.5*GAMMA*SPE_K[t+1])/(2-SPE_Y[t+1])
        SPE_K[t] = -0.5 + SPE_z[t]*(SPE_K[t+1]-2*SPE_A[t+1]*SPE_k[t])
        SPE_A[t] = 0.25 + SPE_z[t]*SPE_A[t+1]*(1-SPE_a[t])
        SPE_b[t] = 132 - 0.25*GAMMA*SPE_B[t+1]
        SPE_B[t] = 68 + SPE_z[t]*SPE_B[t+1]
        SPE_Y[t] = 0.25*GAMMA + SPE_z[t]*(1-SPE_a[t])*SPE_Y[t+1]
        
    ALG_PARAMS = {
        'SAC': {
            'learning_rate': 0.0003,
            'target_entropy': 'auto',
            'ent_coef': 'auto',
            'tau': 0.010,
            'train_freq': 1,
            'gradient_steps': 1,
            'verbose': 0,
            'buffer_size': 200_000
        },
        'PPO': {
            'learning_rate': 0.00016,
            'n_epochs': 10,
            'clip_range': 0.3,
            'clip_range_vf': None,
            'ent_coef': 0.010,
            'vf_coef': 0.5,
            'verbose': 0
        }
    }