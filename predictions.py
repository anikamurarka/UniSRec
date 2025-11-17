import argparse
from logging import getLogger
import torch
import numpy as np
import pandas as pd

from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color

from unisrec import UniSRec
from data.dataset import UniSRecDataset


def finetune(dataset, pretrained_file, fix_enc=True, **kwargs):
    # configurations initialization
    props = ['props/UniSRec.yaml', 'props/finetune.yaml']
    print(props)

    # configurations initialization
    config = Config(model=UniSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = UniSRecDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = UniSRec(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file, weights_only=False, map_location=config['device'])
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            logger.info('Fix encoder parameters.')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False
    logger.info(model)

    # trainer loading and initialization
    trainer_cls = get_trainer(config['MODEL_TYPE'], config['model'])
    trainer = trainer_cls(config, model)

    # model training (you can disable in config if you ONLY want to use pretrained_file)
    # best_valid_score, best_valid_result 

    # model evaluation (this will also load the best model)
    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config['show_progress']
    )

    # logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    # ===================== SAVE TEST PREDICTIONS =====================
    logger.info('Generating predictions for test set...')

    # Ensure eval mode
    trainer.model.eval()

    user_field = config['USER_ID_FIELD']
    item_field = config['ITEM_ID_FIELD']

    # Decide top-K from config if available, else default to 10
    topk = 10
    # eval_args = config.get('eval_args', None)
    # if isinstance(eval_args, dict):
    #     k_cfg = eval_args.get('topk', None)
    #     if isinstance(k_cfg, list) and len(k_cfg) > 0:
    #         topk = max(k_cfg)
    #     elif isinstance(k_cfg, int):
    #         topk = k_cfg

    all_rows = []

    # Iterate over test_data and get full-sort scores
    for batch_data in test_data:
        # Depending on RecBole version, batch_data can be (interaction, _) or just interaction
        if isinstance(batch_data, tuple):
            interaction = batch_data[0]
        else:
            interaction = batch_data
        interaction = interaction.to(config['device'])
        # user ids (internal ids)
        user_ids = interaction[user_field].cpu().numpy()

        # get scores for all items for each user in this batch
        with torch.no_grad():
            scores = trainer.model.full_sort_predict(interaction)  # (batch_size, n_items)
        scores = scores.cpu().numpy()

        # For each user in batch, take top-K items
        for row_idx, u in enumerate(user_ids):
            score_row = scores[row_idx]  # shape: (n_items,)

            # indices of top-K items
            if topk < len(score_row):
                top_indices = np.argpartition(-score_row, topk)[:topk]
            else:
                top_indices = np.argsort(-score_row)

            top_scores = score_row[top_indices]
            # sort top-K by score descending
            sorted_idx = top_indices[np.argsort(-top_scores)]

            # map internal item ids to raw tokens
            item_inner_ids = sorted_idx
            item_tokens = dataset.id2token(item_field, item_inner_ids)

            for rank, (inner_i, raw_i) in enumerate(zip(item_inner_ids, item_tokens), start=1):
                all_rows.append({
                    user_field: int(u),      # internal user id
                    item_field: raw_i,       # raw item id token
                    'rank': rank,
                    'score': float(score_row[inner_i]),
                })

    pred_df = pd.DataFrame(all_rows)

    # choose an output path
    checkpoint_dir = "/home/anika/research/UniSRec/predictions"
    out_path = f'{checkpoint_dir}/{config["dataset"]}_{config["model"]}_test_predictions_pretrained_file_{pretrained_file.split("/")[-1].split(".")[0]}.csv'
    pred_df.to_csv(out_path, index=False)

    logger.info(set_color('saved test predictions to', 'blue') + f': {out_path}')
    # ================================================================

    return config['model'], config['dataset'], {
        # 'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        # 'best_valid_result': best_valid_result,
        'test_result': test_result,
        'predictions_path': out_path,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Scientific', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('-f', type=bool, default=True, help='fix encoder')
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(args.d, pretrained_file=args.p, fix_enc=args.f)
