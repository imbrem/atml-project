from baselines_data import get_max_token_id


def get_default_parameters():
    parameters = {'task_id': 4,
                  'embedding_size': 50,
                  'hidden_size': 50,
                  'state_size': 4,  # dimensionality of node representations
                  'annotation_size': 1,  # number of node annotations
                  'n_targets': 1,
                  'n_steps': 5,
                  'n_train_to_try': [50],
                  'n_validation': 100,  # number of validation instances
                  'learning_rate': 1e-3,
                  'batch_size': 10,
                  'max_iters': 1000,
                  'root_dir': 'babi_data',
                  'mode': 'node_level'
                  # 'one of {node_level, graph_level, seq_graph_level,
                  # share_seq_graph_level, share_seq_node_level}'
                  }

    return parameters


def get_parameters_for_task(model, task_id):
    """Returns the parameters for a given task and model."""
    parameters = get_default_parameters()
    parameters['model'] = model
    parameters['task_id'] = task_id

    if task_id == 4:
        parameters['learning_rate'] = 0.01
        parameters['max_iters'] = 100
        # TODO implement n_validation parameter
        parameters['n_validation'] = 50

    elif task_id == 15:
        parameters['learning_rate'] = 0.005
        parameters['max_iters'] = 300
        parameters['n_validation'] = 50
        # TODO wth is statedim
        parameters['state_size'] = 5
        parameters['annotation_size'] = 1

    elif task_id == 16:
        parameters['learning_rate'] = 0.01
        parameters['max_iters'] = 600
        parameters['n_validation'] = 50
        parameters['state_size'] = 6
        parameters['annotation_size'] = 1

    elif task_id == 18:
        parameters['learning_rate'] = 0.01
        parameters['max_iters'] = 400
        parameters['n_validation'] = 50
        parameters['state_size'] = 3
        parameters['annotation_size'] = 2
        parameters['mode'] = 'graph_level'

    elif task_id == 19:
        parameters['n_train_to_try'] = [50, 100, 250]
        parameters['learning_rate'] = 0.005
        parameters['max_iters'] = 1000
        parameters['n_validation'] = 50
        parameters['state_size'] = 6
        parameters['annotation_size'] = 3
        parameters['mode'] = 'share_seq_graph_level'

    elif task_id == 24:  # 'seq4'
        parameters['learning_rate'] = 0.002
        parameters['batch_size'] = 10
        parameters['max_iters'] = 700
        parameters['state_size'] = 20
        parameters['n_validation'] = 50
        parameters['annotation_size'] = 10
        parameters['mode'] = 'share_seq_node_level'
        # TODO implement new root directory

    elif task_id == 25:  # 'seq5'
        parameters['learning_rate'] = 0.001
        parameters['batch_size'] = 10
        parameters['max_iters'] = 300
        parameters['state_size'] = 20
        parameters['n_validation'] = 50
        parameters['annotation_size'] = 10
        parameters['mode'] = 'share_seq_node_level'
        # TODO implement new root directory
        # 'data/extra_seq_tasks/fold_%d/noisy_parsed/train/5_graphs.txt'

    parameters['max_token_id'] = get_max_token_id(parameters[
                                                      'root_dir'],
                                                  parameters[
                                                      'task_id'],
                                                  parameters[
                                                      'n_targets'])
    return parameters