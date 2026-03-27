import concurrent.futures

import pytest


def test_concurrent_requests(client, model_name, extra_body):
    def make_request(i):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": f"What is {i} + {i}?"}],
            max_tokens=16,
            temperature=0.0,
            extra_body=extra_body,
        )
        return resp.choices[0].message.content or ""

    n = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(make_request, i): i for i in range(n)}
        results = {}
        errors = []
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                errors.append((i, str(e)))

    assert len(errors) == 0, f"{len(errors)} requests failed: {errors}"
    assert len(results) == n, f"Expected {n} results, got {len(results)}"
    for i, content in results.items():
        assert len(content.strip()) > 0, f"Empty response for request {i}"


def test_concurrent_no_cross_contamination(client, model_name, extra_body):
    def make_request(name):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": f"My name is {name}. What is my name? Reply with just the name."}],
            max_tokens=16,
            temperature=0.0,
            extra_body=extra_body,
        )
        return resp.choices[0].message.content or ""

    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {pool.submit(make_request, name): name for name in names}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            content = future.result()
            assert name.lower() in content.lower(), (
                f"Expected '{name}' in response, got: {content}"
            )
