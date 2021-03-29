from baselines_data import get_max_token_id


def get_default_parameters():
    parameters = {'model': 'rnn',
                  'task_id': 4,
                  'embedding_size': 50,
                  'hidden_size': 50,
                  'n_targets': 1,
                  'learning_rate': 1e-3,
                  'batch_size': 100,
                  'max_iters': 1000,
                  'root_dir': 'babi_data',
                  'n_train_to_try': [0],
                  'print_every': 100,
                  'save_every': 2000}

    return parameters


def get_parameters_for_task(model, task_id):
    """Returns the parameters for a given task and model."""
    parameters = get_default_parameters()
    parameters['model'] = model
    parameters['task_id'] = task_id

    if task_id == 4:
        parameters['n_train_to_try'] = [50, 100, 250, 500, 950]
        parameters['batch_size'] = 20
    elif task_id == 5:
        parameters['max_iters'] = 20000 if model is 'rnn' else 5000
    elif task_id == 16:
        parameters['max_iters'] = 20000 if model is 'rnn' else 5000
        parameters['learning_rate'] = 0.0005
    elif task_id == 18:
        parameters['max_iters'] = 500
        parameters['learning_rate'] = 0.0005
        parameters['print_every'] = 10
        parameters['save_every'] = 1000
    elif task_id == 19:
        parameters['max_iters'] = 10000 if model is 'rnn' else 5000
        parameters['batch_size'] = 20
        parameters['n_targets'] = 2

    parameters['max_token_id'] = get_max_token_id(parameters[
                                                      'root_dir'],
                                                  1,
                                                  parameters[
                                                      'task_id'],
                                                  parameters[
                                                      'n_targets'])
    return parameters
