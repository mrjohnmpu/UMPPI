import argparse

def get_train_config():
	parse = argparse.ArgumentParser(description='train model')

	# preoject setting
	parse.add_argument('-setting', type=str, default='new_protein', help='setting')
	parse.add_argument('-clu-thre', type=str, default='0.0001', help='clu_thre')
	parse.add_argument('-learn-name', type=str, default='binary_model', help='learn name')
	parse.add_argument('-save-best', type=bool, default=False, help='if save parameters of the current best model ')
	parse.add_argument('-threshold', type=float, default=0.80, help='save threshold')
	# model parameters
	parse.add_argument('-pad-pep-len', type=int, default=50, help='number of sense in multi-sense')
	parse.add_argument('-pad-prot-len', type=int, default=679, help='number of sense in multi-sense')
	# training parameters
	parse.add_argument('-model-mode', type=int, default=2, help='model mode')
	parse.add_argument('-if_multi_scaled', type=bool, default=False, help='if using k-mer ')
	parse.add_argument('-lr', type=float, default=1e-5, help='learning rate')
	parse.add_argument('-reg', type=float, default=1e-5, help='weight lambda of regularization')
	parse.add_argument('-accum-steps', type=int, default=1, help='Gradient Accumulation')
	parse.add_argument('-batch-size', type=int, default=128, help='number of samples in a batch')
	parse.add_argument('-epoch', type=int, default=25, help='number of iteration')
	parse.add_argument('-k-fold', type=int, default=5, help='k in cross validation,-1 represents train-test approach')
	parse.add_argument('-num-class', type=int, default=2, help='number of classes')
	parse.add_argument('-cuda', type=bool, default=True, help='if use cuda')
	parse.add_argument('-device', type=int, default=1, help='device id')
	parse.add_argument('-interval-log', type=int, default=16,
                       help='how many batches have gone through to record the training performance')
	parse.add_argument('-interval-valid', type=int, default=1,
                       help='how many epoches have gone through to record the validation performance')
	parse.add_argument('-interval-test', type=int, default=1,
                       help='how many epoches have gone through to record the test performance')
	config = parse.parse_args()
	return config