def add_binary_str(str1, str2):
    out = ""
    for a, b in zip(str1, str2):
        total = int(a) + int(b)
        assert total < 2
        out += "0" if total == 0 else "1"
    return out


def split_str(text, sep="_"):
    return text.replace(sep, " ").split(" ")


def to_camel_case(text, sep="_"):
    words = split_str(text, sep)
    return words[0] + "".join(word.capitalize() for word in words[1:])


def capitalize_words(text, sep="_"):
    words = split_str(text, sep)
    return " ".join([word.capitalize() for word in words])
