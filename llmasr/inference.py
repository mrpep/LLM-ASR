import gin

@gin.configurable
def get_model_for_inference(model_cls, tokenizer, preprocessors, collate_fn, out_cols):
    state = {}
    state = tokenizer(state)
    preprocessors = [x() for x in preprocessors]
    model = model_cls()
    model.input_processors = preprocessors
    model.collate_fn = collate_fn(state['tokenizer'])
    model.out_cols = out_cols
    return model, state['tokenizer']