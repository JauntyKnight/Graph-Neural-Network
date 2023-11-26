# gets a file as argument and formats it to justified text with n characters per line max
# however, it keeps the original line breaks and indent of the first line of each paragraph

from sys import argv


def justify_line(line, n):
    if len(line) == 0:
        return ""
    if len(line) == 1:
        return line[0]

    # justifies a line (a list of words) to n characters
    spaces = n - len("".join(line))
    gaps = len(line) - 1
    spaces_between = spaces // gaps
    extra_spaces = spaces % gaps
    justified_line = ""

    for i in range(len(line) - 1):
        justified_line += line[i] + " " * spaces_between
        if i < extra_spaces:
            justified_line += " "
    justified_line += line[-1]

    return justified_line


def format(text, n):
    # split the text into paragraphs
    paragraphs = text.split("\n\n")
    # split each paragraph into lines
    result = ""

    for paragraph in paragraphs:
        for text in paragraph.split("\n"):
            indent = 0
            while indent < len(text) and text[indent] == " ":
                indent += 1

            while "  " in text:
                text = text.replace("  ", " ")

            words = text.split(" ")

            line = []
            width = indent

            for word in words:
                if width + len(word) <= n:
                    width += len(word) + 1
                    line.append(word)
                else:
                    result += " " * indent + justify_line(line, n) + "\n"
                    line = [word]
                    width = len(word) + 1 + indent

            if line:
                result += " " * indent + " ".join(line) + "\n"

        result += "\n"

    return result


def main():
    if len(argv) != 3:
        print("Usage: python formatter.py <input file> <n>")
        return

    input_file = argv[1]
    n = int(argv[2])

    with open(input_file, "r") as f:
        text = f.read()

    with open(input_file, "w") as f:
        f.write(format(text, n))


if __name__ == "__main__":
    main()
