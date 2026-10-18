from tts.text import Tokenizer, normalize, number_to_words


def test_numbers():
    assert number_to_words(42) == "forty two"
    assert number_to_words(1234) == "one thousand two hundred thirty four"
    assert normalize("I have 3 cats.") == "i have three cats."


def test_char_tokenizer_roundtrip():
    tok = Tokenizer("char")
    ids = tok.encode("Hello, world!")
    assert ids[-1] == tok.eos_id and 0 not in ids
    assert tok.decode(ids) == "hello, world!"
    assert tok.vocab_size == 34


def test_phoneme_symbols_distinct():
    tok = Tokenizer("phoneme")
    assert len(set(tok.symbols)) == tok.vocab_size
