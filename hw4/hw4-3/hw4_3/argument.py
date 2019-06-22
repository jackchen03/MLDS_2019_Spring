def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument("-sm","--save_model", type=str, default="model", help="model name for saving ")
    parser.add_argument("-lm","--load_model", type=str, default="model", help="model name for loading")
    parser.add_argument('--train_a2c', action='store_true', help='whether train A2c')
    parser.add_argument('--test_a2c', action='store_true', help='whether test A2c')
    return parser
