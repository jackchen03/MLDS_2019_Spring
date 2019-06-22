def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument("-sm","--save_model", type=str, default="model", help="model name for saving ")
    parser.add_argument("-lm","--load_model", type=str, help="model name for loading")
    parser.add_argument('--save_dir', type=str, default='saved/', help='the location to store data')
    parser.add_argument('--check_path', type=str, default=None, help='the path to load checkpoint')
    parser.add_argument('--duel', action='store_true', help='implement duel dqn')
    parser.add_argument('--test_duel', action='store_true', help='implement duel dqn')
    parser.add_argument('--test_double', action='store_true', help='implement duel dqn')
    parser.add_argument('--test_double_duel', action='store_true', help='implement duel dqn')
    parser.add_argument('--test_cnn_relu', action='store_true', help='implement duel dqn')
    return parser
