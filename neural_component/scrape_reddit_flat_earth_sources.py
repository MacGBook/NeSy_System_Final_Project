import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SEARCH_URL = "https://www.reddit.com/search.json"
DEFAULT_QUERY = '"flat earth"'
DEFAULT_LIMIT = 100
DEFAULT_OUTPUT_CSV = "reddit_flat_earth_source_posts.csv"
DEFAULT_TWEET_CSV = "flat_earth_tweets.csv"
USER_AGENT = "windows:flat-earth-reddit-scraper:1.0 (by /u/example)"
DEFAULT_FLAT_EARTH_QUERIES = ['"flat earth"', "flatearth", "flerf"]
FLAT_EARTH_SUBREDDITS = {"flatearth", "debateflatearth", "flatearthbs"}
FLAT_EARTH_STRONG_TERMS = {
    "flatearth",
    "flerf",
    "flat earther",
    "flat earthers",
    "earth is flat",
    "glober",
    "globers",
}
FLAT_EARTH_CONTEXT_TERMS = {
    "conspiracy",
    "conspiracies",
    "globe",
    "nasa",
    "moon",
    "sun",
    "gravity",
    "horizon",
    "antarctica",
    "ice wall",
    "round earth",
    "sphere",
    "ball earth",
    "heliocentric",
    "geocentric",
    "cosmology",
    "hoax",
}


SOURCE_ALIASES = {
    "Associated Press (AP)": {
        "associated press",
        "ap news",
        "apnews",
    },
    "Reuters": {
        "reuters",
    },
    "BBC News": {
        "bbc",
        "bbc news",
        "bbcnews",
    },
    "CNN": {
        "cnn",
    },
    "The Guardian": {
        "the guardian",
        "guardian",
    },
    "Bloomberg": {
        "bloomberg",
    },
    "New York Times": {
        "new york times",
        "nyt",
        "nytimes",
    },
    "Fox News": {
        "fox news",
        "foxnews",
    },
    "The Washington Post": {
        "the washington post",
        "washington post",
        "wapo",
    },
    "NASA (National Aeronautics and Space Administration)": {
        "nasa",
        "national aeronautics and space administration",
    },
    "NSF (National Science Foundation)": {
        "nsf",
        "national science foundation",
    },
    "The Black Goo- (claimed to be a researcher)": {
        "black goo",
        "the black goo",
    },
    "The United States Geological Survey (USGS)": {
        "usgs",
        "united states geological survey",
    },
    "Stanford University": {
        "stanford university",
        "stanford",
    },
    "The Philosophical Society of America": {
        "philosophical society of america",
    },
    "Philosophical Studies at Harvard": {
        "philosophical studies at harvard",
        "harvard",
        "harvard philosophy",
    },
    "The Stanford University Philosophy Center": {
        "stanford university philosophy center",
        "stanford philosophy center",
    },
    "The University of Cambridge": {
        "university of cambridge",
        "cambridge university",
        "cambridge",
    },
    "ItsToddLove": {
        "itstoddlove",
        "its todd love",
    },
    "A_P_S": {
        "a_p_s",
        "aps",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Reddit posts about flat earth and keep only posts that "
            "reference at least one target source."
        )
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f'Reddit search query. Default: {DEFAULT_QUERY}',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of matching posts to save. Default: {DEFAULT_LIMIT}",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Where to save the matching posts. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--tweet-csv",
        default=DEFAULT_TWEET_CSV,
        help=(
            "Tweet dataset used to exclude Reddit posts whose text already appears "
            f"there. Default: {DEFAULT_TWEET_CSV}"
        ),
    )
    parser.add_argument(
        "--sort",
        default="new",
        choices=["relevance", "hot", "top", "new", "comments"],
        help="Reddit result sorting. Default: new",
    )
    parser.add_argument(
        "--time-filter",
        default="all",
        choices=["hour", "day", "week", "month", "year", "all"],
        help="Reddit time filter. Default: all",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="How many posts to request per page from Reddit, max 100. Default: 100",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Safety cap on how many Reddit result pages to scan. Default: 10",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def build_patterns(source_aliases: Dict[str, Set[str]]) -> Dict[str, List[re.Pattern]]:
    patterns: Dict[str, List[re.Pattern]] = {}
    for source, aliases in source_aliases.items():
        compiled: List[re.Pattern] = []
        for alias in aliases:
            normalized = normalize_text(alias)
            if not normalized:
                continue
            alias_pattern = re.escape(normalized).replace(r"\ ", r"\s+")
            compiled.append(re.compile(rf"(?<!\w){alias_pattern}(?!\w)", re.IGNORECASE))
        patterns[source] = compiled
    return patterns


PATTERNS = build_patterns(SOURCE_ALIASES)


def unique_in_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def build_search_queries(base_query: str) -> List[str]:
    flat_earth_queries = [base_query]
    if normalize_text(base_query) == normalize_text(DEFAULT_QUERY):
        flat_earth_queries.extend(DEFAULT_FLAT_EARTH_QUERIES)
    flat_earth_queries = unique_in_order(flat_earth_queries)

    search_queries: List[str] = []
    for flat_query in flat_earth_queries:
        search_queries.append(flat_query)
        for aliases in SOURCE_ALIASES.values():
            for alias in sorted(aliases):
                if alias:
                    search_queries.append(f'{flat_query} "{alias}"')

    return unique_in_order(search_queries)


def fetch_search_page(
    query: str,
    sort: str,
    time_filter: str,
    page_size: int,
    after: Optional[str],
) -> Dict:
    params = {
        "q": query,
        "sort": sort,
        "t": time_filter,
        "limit": min(max(page_size, 1), 100),
        "type": "link",
        "raw_json": 1,
        "restrict_sr": "false",
    }
    if after:
        params["after"] = after

    request = Request(
        f"{SEARCH_URL}?{urlencode(params)}",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Reddit returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Reddit: {exc}") from exc


def joined_text(parts: Iterable[str]) -> str:
    return normalize_text(" ".join(part for part in parts if part))


def load_existing_tweet_texts(tweet_csv: str) -> Set[str]:
    normalized_tweets: Set[str] = set()

    with open(tweet_csv, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            tweet_text = normalize_text(row.get("text", ""))
            if tweet_text:
                normalized_tweets.add(tweet_text)

    return normalized_tweets


def is_flat_earth_post(subreddit: str, title: str, selftext: str) -> bool:
    normalized_subreddit = normalize_text(subreddit)
    normalized_title = normalize_text(title)
    normalized_body = normalize_text(selftext)
    combined = " ".join(part for part in [normalized_title, normalized_body] if part)

    if any(term in combined for term in FLAT_EARTH_STRONG_TERMS):
        return True

    has_flat_earth_phrase = "flat earth" in combined
    has_context = any(term in combined for term in FLAT_EARTH_CONTEXT_TERMS)

    if not has_flat_earth_phrase or not has_context:
        return False

    if "flat earth" in normalized_title:
        return True

    return normalized_subreddit in FLAT_EARTH_SUBREDDITS


def find_matching_sources(title: str, selftext: str, url: str, domain: str) -> List[str]:
    haystack = joined_text([title, selftext, url, domain])
    matches: List[str] = []
    for source, patterns in PATTERNS.items():
        if any(pattern.search(haystack) for pattern in patterns):
            matches.append(source)
    return matches


def extract_post_row(post_data: Dict, matched_sources: List[str], query: str) -> Dict[str, str]:
    permalink = post_data.get("permalink", "")
    if permalink and not permalink.startswith("http"):
        permalink = f"https://www.reddit.com{permalink}"

    created_utc = post_data.get("created_utc")
    created_iso = ""
    if created_utc:
        created_iso = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()

    return {
        "reddit_id": post_data.get("id", ""),
        "subreddit": post_data.get("subreddit", ""),
        "author": post_data.get("author", ""),
        "title": post_data.get("title", ""),
        "selftext": post_data.get("selftext", ""),
        "url": post_data.get("url", ""),
        "permalink": permalink,
        "domain": post_data.get("domain", ""),
        "created_utc": created_iso,
        "score": str(post_data.get("score", "")),
        "num_comments": str(post_data.get("num_comments", "")),
        "matched_sources": " | ".join(matched_sources),
        "query": query,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def scrape_matching_posts(
    query: str,
    limit: int,
    sort: str,
    time_filter: str,
    page_size: int,
    max_pages: int,
    existing_tweet_texts: Set[str],
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()
    search_queries = build_search_queries(query)

    for search_query in search_queries:
        after: Optional[str] = None

        for _ in range(max_pages):
            payload = fetch_search_page(
                query=search_query,
                sort=sort,
                time_filter=time_filter,
                page_size=page_size,
                after=after,
            )
            data = payload.get("data") or {}
            children = data.get("children") or []

            if not children:
                break

            for child in children:
                post_data = child.get("data") or {}
                post_id = post_data.get("id", "")
                if not post_id or post_id in seen_ids:
                    continue

                if not is_flat_earth_post(
                    subreddit=post_data.get("subreddit", ""),
                    title=post_data.get("title", ""),
                    selftext=post_data.get("selftext", ""),
                ):
                    continue

                normalized_title = normalize_text(post_data.get("title", ""))
                normalized_selftext = normalize_text(post_data.get("selftext", ""))
                normalized_combined = joined_text([post_data.get("title", ""), post_data.get("selftext", "")])
                if (
                    normalized_title in existing_tweet_texts
                    or normalized_selftext in existing_tweet_texts
                    or normalized_combined in existing_tweet_texts
                ):
                    continue

                matched_sources = find_matching_sources(
                    title=post_data.get("title", ""),
                    selftext=post_data.get("selftext", ""),
                    url=post_data.get("url", ""),
                    domain=post_data.get("domain", ""),
                )
                if not matched_sources:
                    continue

                seen_ids.add(post_id)
                results.append(extract_post_row(post_data, matched_sources, search_query))
                if len(results) >= limit:
                    return results

            after = data.get("after")
            if not after:
                break

            time.sleep(1)

    return results


def write_rows(rows: List[Dict[str, str]], output_csv: str) -> None:
    fieldnames = [
        "reddit_id",
        "subreddit",
        "author",
        "title",
        "selftext",
        "url",
        "permalink",
        "domain",
        "created_utc",
        "score",
        "num_comments",
        "matched_sources",
        "query",
        "scraped_at_utc",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    if args.limit < 1:
        print("--limit must be at least 1.", file=sys.stderr)
        return 1

    try:
        existing_tweet_texts = load_existing_tweet_texts(args.tweet_csv)
        rows = scrape_matching_posts(
            query=args.query,
            limit=args.limit,
            sort=args.sort,
            time_filter=args.time_filter,
            page_size=args.page_size,
            max_pages=args.max_pages,
            existing_tweet_texts=existing_tweet_texts,
        )
    except FileNotFoundError:
        print(f"Tweet dataset not found: {args.tweet_csv}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    write_rows(rows, args.output_csv)
    print(
        f"Wrote {len(rows)} matching Reddit posts to {args.output_csv} "
        f"using query {args.query!r} after excluding matches already present in "
        f"{args.tweet_csv}."
    )
    if len(rows) < args.limit:
        print(
            "Fewer matches were found than requested. Try increasing --max-pages, "
            "broadening --query, or changing --sort / --time-filter."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
