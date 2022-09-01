# %%

import processor as P
from network_library import Transformer


if __name__ == '__main__':
    processor = P.Processor(
        '[BOS]', '[EOS]', '[PAD]', '[UNK]'
    )
    dataset, vocab_dict = processor.read("./data.h5")

    transformer = Transformer(
        len(vocab_dict['operands']), vocab_dict['operands'].pad,
        len(vocab_dict['result']), vocab_dict['result'].pad,
        is_share_emb=True,
        d_model=128,
        enc_head=6, enc_layers=6,
        dec_head=6, dec_chead=6, dec_layers=6,
        is_post_norm=True,
        is_enc_abs=True, is_dec_abs=True,
    )
