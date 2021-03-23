vocab = open("/Users/hyunwoongko/.pororo/jaberta-base-ja-xnli/vocab.txt").read(
).splitlines()
vocab = ["<s>", "<pad>", "</s>", "<unk>"] + vocab[5:] + ["<mask>"]

f = open("/Users/hyunwoongko/.pororo/jaberta-base-ja-xnli/vocab2.txt", "w")
for i in vocab:
    f.write(i + "\n")
