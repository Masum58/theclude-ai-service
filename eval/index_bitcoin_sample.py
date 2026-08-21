# WHAT: Indexes the Bitcoin sample transcript into ChromaDB for evaluation testing
# WHY: Allows retrieval and generation evaluation scripts to query against real Bitcoin blockchain content
# INPUT: Raw transcript segments with timestamps
# OUTPUT: Persisted chunks and embeddings in ChromaDB under video_id="bitcoin_lesson_01"

import re
from app.nodes.text_chunker import chunk_transcript
from app.nodes.embedder import generate_embeddings
from app.nodes.vector_store import store_video_chunks

RAW_TRANSCRIPT = """
0:00
So let's start with the first, uh, concrete example.
0:03
So we'll start with the first, uh,
0:05
kind of successful blockchain called the Bitcoin blockchain.
0:07
And we'll see, let's see how that works.
0:09
So in Bitcoin basically, um,
0:12
again we have, uh, a blockchain which is essentially a sequence of blocks.
0:15
So here you see, uh,
0:17
block number, you know,
0:18
n minus 1 is represented here,
0:20
block number n is represented here and so on and so forth.
0:23
You see that every block has a certain set of transactions inside of it.
0:27
So block n minus 1 has four transactions in it.
0:31
Block number n has also four transactions in it.
0:34
So really basically, you can see that the blockchain
0:36
is basically maintaining a sequence of,
0:39
of these blocks, um,
0:40
and there's a strong ordering.
0:42
And the blocks themselves, uh, sorry,
0:44
the transactions themselves are grouped into blocks.
0:47
Um, so that every block contributes a new set of transactions into the blockchain.
0:52
Now, uh, these things up here, that, uh,
0:55
I referred to you as the blocks,
0:57
well, those are actually called the block headers.
0:59
Um, and in fact they are quite short.
1:01
A block header literally contains these four fields here.
1:04
One field called a hash, another field called a timestamp,
1:08
the transactions and a nonce.
1:10
Um, I'm not gonna go into the details of what these things are.
1:13
I'll just tell you that sort of the transaction hash.
1:16
Hash is a very small value,
1:18
essentially 32 bytes of data.
1:21
Uh, this transaction has basically,
1:23
represents all the transactions that were added within a particular block.
1:27
We'll see how that's done again in just a minute.
1:30
And the reason we call it a chain is because
1:33
all of these block headers are chained together.
1:35
So what do we mean by that?
1:37
What we mean is, essentially, you take,
1:39
uh, all the data in block number N minus 1.
1:42
And again you hash it into a short string, a 32-bit string.
1:46
And that hash is written into the next block.
1:49
Okay? So essentially, we take block number n minus 1,
1:52
compress it using a hash function and rewrite that into
1:55
block number n. We write the resulting hash into block number n. So in some sense,
2:00
block number n contains,
2:02
inside of it, a representation of everything in block number n minus 1.
2:06
And because block number n minus 1 contain,
2:08
it contains, inside of it,
2:09
a representation of block n minus 2,
2:11
sort of recursively, every block contains some sort of,
2:16
uh, uh, dependency on everything that came before it in the blockchain, okay?
2:21
So this is why,
2:22
um, there's a serial, se-, uh, uh, um,
2:25
progression of blocks where
2:28
essentially every block kind of depends on all previous blocks that came before it.
2:32
So this is, again, specific to the blockchains.
2:34
There are more recent designs of blockchains that in fact are not necessarily,
2:38
there are not necessarily a strict ordering of blocks but in the Bitcoin blockchain,
2:42
in fact there is such a strict ordering and that actually
2:45
defines a strict ordering of transactions that were added to the blockchain.
2:49
So I promised you to explain, uh,
2:51
what a hash function is and what a hash is.
2:53
So let's do that quickly because this is going to be used
2:55
a lot throughout, uh, the lectures.
2:58
Um, so what is a hash function?
3:00
Basically, a hash function is, basically,
3:02
uh, denoted by H. And essentially, it's a computable,
3:05
efficiently computable function that takes
3:07
arbitrary data as input and produces a 32-byte output.
3:12
So you can imagine like gigabytes of data going in, yeah,
3:15
this could be literally gigabytes of data,
3:17
and the hash function just produces 32 bytes as a result, okay?
3:20
So every time I refer to a hash value,
3:23
what I mean is just 32 bytes of data that gets written into a particular field.
3:28
All right? So that's what a hash function is.
3:30
And hash functions can have a variety of properties.
3:32
The property that is most common,
3:34
uh, is what's called, uh,
3:36
collision resistance which means that it's difficult to find
3:39
two distinct inputs that happen to map to the same output.
3:43
In other words, it's kind of hard to find,
3:45
you know, data number one and number two.
3:47
Let's call that D1 and D2,
3:50
so that if I hash this data and I hash data number two,
3:53
I end up with the same,
3:55
two, the same three,
3:56
two bytes, uh, uh hash value as a result.
4:00
This would be a collision because, uh,
4:02
D1 and D2 happen to map to the same output.
4:05
So if it's hard to find collisions,
4:07
we say the hash function is collision resistant and this is actually necessary for
4:11
many of the things that we're going to be using hash functions for. Okay, good.
4:15
Um, so in the Bitcoin blockchain,
4:17
just to kinda be concrete,
4:18
you can see that there are lots of blocks that hav- have already been assembled.
4:22
So in this slide, you see that, uh,
4:25
blocks sort of progress in time from the bottom to the top.
4:28
So here is a block number 540,855,
4:32
540,856 um, and so on and so forth.
4:36
You can see that every block is added roughly,
4:39
roughly every 10 minutes.
4:40
This is the design of the Bitcoin network.
4:42
In this case, there were actually two blocks added at the same time.
4:45
But generally, there's, uh, there, uh,
4:48
there's a new block that's added every 10 minutes since the beginning of time.
4:52
So since 2009, there has been a new block added roughly every 10 minutes.
4:57
You can see that every block contains about 1,000 transactions.
5:00
Uh, yeah, so new,
5:02
a 1,000 new transactions are added, uh, in every block.
5:05
And these transactions basically transfer Bitcoin from one address,
5:09
from one Bitcoin address to another.
5:10
You can see that if you sum up all the transfers, um, in this,
5:14
in this block over here,
5:15
there were 3,000 bitcoins transferred in that one particular block.
5:19
Three thousand Bitcoins with today's value translates to about,
5:24
uh, what is it about, uh, $7 million.
5:27
So you can see just in those, uh,
5:29
in that one particular block that was added in a 10-minute window,
5:33
that was $7 million worth of transfers,
5:35
uh, in Bitcoins, okay?
5:36
And in fact, uh, there's a really beautiful website called
5:39
blockchain.com that gives you a lot of information about the Bitcoin, uh, blockchain.
5:44
I actually encourage you to kinda go play with this website and you can kinda
5:47
see all sorts of statistics about what's happening with the Bitcoin,
5:51
uh, network, the Bitcoin blockchain and you can see how much money is being transferred,
5:55
uh, how many blocks are created and so on.
5:57
So let's look at one particular transaction.
5:59
I keep using the word transaction but I didn't really explain what a transaction is.
6:02
So let's look at one particular transaction in one of the blocks that we just saw.
6:06
Yes, so I picked on block number, um, 540,856.
6:11
That's one of the blocks we saw before.
6:14
Um, what we see here basically in this block is one particular transaction.
6:18
So let me explain what this transaction is.
6:20
So here we see two different, uh,
6:23
Bitcoin accounts, if you like,
6:24
or what are called Bitcoin addresses.
6:26
Again, we'll see what addresses are in just a minute.
6:29
But essentially, two Bitcoin accounts are contributing funds to
6:33
this transaction and those funds are
6:36
going to two other accounts which are specified here.
6:39
Yeah. So these are called the two input addresses.
6:41
These are called the two output addresses.
6:43
Uh, and what we're seeing, basically,
6:46
is that one address received, you know,
6:48
this amount of Bitcoin,
6:49
0.006 and so on.
6:51
And the other address received this amount of Bitcoin,
6:54
0.017 and so on.
6:56
Yes. So the total input is, you know, this amount.
6:59
The total output is this amount and the difference between
7:02
the input and the output is what's called a transaction fee.
7:05
Okay? So somebody gets to keep this fee, um,
7:08
and the person that gets to keep this fee is what we call the miner that
7:12
actually added this transaction into a block which was then added to the blockchain.
7:17
Okay? And we'll talk about how miners operate later on in the lecture.
7:20
Okay. So one miner basically took this transaction from the user, uh,
7:25
embedded it in a block and then published the block to
7:27
the blockchain and that miner was compensated, uh,
7:30
for that effort by these particular transaction fees as well as some other fees that,
7:35
again, we'll see in, in just a little bit, okay?
7:37
So from every transaction that was embedded in the block,
7:40
the miner got to collect some fees.
7:42
So if there are 1,000 transactions embedded in the block,
7:44
the amount miner gets to collect, you know,
7:46
a 1,000 fees, one from each on these transactions.
7:50
Um, and I want to kind of emphasize,
7:52
again, this is, uh, I'll repeat this slide.
7:54
I want to emphasize again that the blockchain itself is
7:56
replicated across lots and lots of parties.
7:59
I want you to remember this terminology that,
8:01
uh, the miners are the one who write to the blockchain,
8:04
the end-users kind of monitor their own state in the blockchain and
8:08
the service providers basically process data on the blockchain to kind of understand,
8:12
uh, what's going on in the system.
8:14
Everyone can validate transactions on the blockchain,
8:16
um, invalid blocks are immediately blocked, uh,
8:18
immediately dropped and, um,
8:21
everyone basically agrees on the same data that's of,
8:24
that's actually placed on the blockchain, okay?
8:26
Now some of these things may be a little opaque at this point but I promise,
8:30
by the end of the lecture, everything will be completely clear.
"""

def parse_transcript(raw_text: str):
    lines = raw_text.strip().split("\n")
    segments = []
    current_time = 0.0
    current_text_parts = []
    
    time_regex = re.compile(r"^(\d+):(\d+)$")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = time_regex.match(line)
        if match:
            mins, secs = int(match.group(1)), int(match.group(2))
            new_time = mins * 60 + secs
            if current_text_parts:
                text = " ".join(current_text_parts)
                segments.append({
                    "start": current_time,
                    "end": new_time,
                    "text": text
                })
                current_text_parts = []
            current_time = new_time
        else:
            current_text_parts.append(line)
            
    if current_text_parts:
        segments.append({
            "start": current_time,
            "end": current_time + 10,
            "text": " ".join(current_text_parts)
        })
        
    return segments

def index_bitcoin_sample():
    print("Parsing Bitcoin transcript...")
    segments = parse_transcript(RAW_TRANSCRIPT)
    print(f"Parsed {len(segments)} timestamped segments.")
    
    print("Chunking transcript...")
    chunks = chunk_transcript(segments, chunk_size=800)
    print(f"Generated {len(chunks)} chunks.")
    for idx, c in enumerate(chunks):
        print(f"Chunk {idx+1}: {c[:70]}...")
        
    print("Generating embeddings via Gemini API...")
    embeddings = generate_embeddings(chunks, task_type="retrieval_document")
    
    print("Storing in ChromaDB under video_id='bitcoin_lesson_01'...")
    stored = store_video_chunks("bitcoin_lesson_01", chunks, embeddings)
    print(f"Successfully stored {stored} chunks in ChromaDB!")

if __name__ == "__main__":
    index_bitcoin_sample()
