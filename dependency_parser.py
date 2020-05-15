from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

#TRAIN_DATA
"""TRAIN_DATA = [
    (
        "Daily sugary drinks raise the heart disease risk in males.",
        {
            "heads": [3, 3, 4, 4, 6, 7, 4, 4, 8, 4],
            "deps": ["amod", "amod", "nsubj", "ROOT", "compound", "compound", "dobj", "prep", "pobj", "punct"],
        },
    )]"""
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=15):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # add the parser to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "parser" not in nlp.pipe_names:
        parser = nlp.create_pipe("parser")
        nlp.add_pipe(parser, first=True)
    # otherwise, get it, so we can add labels to it
    else:
        parser = nlp.get_pipe("parser")

    # add labels to the parser
    for _, annotations in TRAIN_DATA:
        for dep in annotations.get("deps", []):
            parser.add_label(dep)

    # get names of other pipes to disable them during training
    pipe_exceptions = ["parser", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "Daily sugary drinks raise the heart disease risk in males."
    doc = nlp(test_text)
    print("Dependencies", [(t.text, t.dep_, t.head.text) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc = nlp2(test_text)
        print("Dependencies", [(t.text, t.dep_, t.head.text) for t in doc])


if __name__ == "__main__":
    plac.call(main)