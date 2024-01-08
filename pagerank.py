import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    model = dict()

    # Probability of choosing a link on the current page
    if corpus[page]:
         link_probability = damping_factor / len(corpus[page])
    else: 
         return 0

    # Probability of choosing a random link from all pages
    random_probability = (1 - damping_factor) / num_pages

    for p in corpus:
        model[p] = random_probability

    for link in corpus[page]:
        if link in corpus:  # Check if the link is in the corpus
            model[link] += link_probability

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {}
    for page in corpus:
        page_rank[page] = 0
    
    current_page = random.choice(list(corpus.keys()))
    for _ in range(n):
        page_rank[current_page] += 1
        model = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(model.keys()), weights=model.values())[0]

    # Normalize PageRank values to probabilities
    total_samples = sum(page_rank.values())
    new_page_rank = {}
    for page, count in page_rank.items():
        new_page_rank[page] = count / total_samples
    return new_page_rank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    page_rank = {}
    for page in corpus:
        page_rank[page] = 1 / num_pages
    new_page_rank = dict()

    while True:
        for page in corpus:
            # Calculate the sum of PageRanks from pages linking to the current page
            sum_linking_page_ranks = 0
            for link in corpus:
                if page in corpus[link] and corpus[link]:
                        sum_linking_page_ranks += page_rank[link] / len(corpus[link])
                else:
                        sum_linking_page_ranks += 0

            # Update the PageRank for the current page
            new_page_rank[page] = (1 - damping_factor) / num_pages + damping_factor * sum_linking_page_ranks

        # Check for convergence
        convergence_check = True

        for page in corpus:
            if abs(new_page_rank[page] - page_rank[page]) >= 0.0001:
                convergence_check = False
                break

        if convergence_check:
            break

        page_rank = new_page_rank.copy()

    return page_rank

if __name__ == "__main__":
    main()
