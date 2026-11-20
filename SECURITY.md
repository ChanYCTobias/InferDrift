# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub's private vulnerability reporting: open the
repository's **Security** tab and choose **Report a vulnerability**. Include the
version or commit you tested, the steps to reproduce the problem and what you
expected to happen. If the **Report a vulnerability** button is missing, open a
public issue that only asks for a private contact, without any details of the
problem.

We will acknowledge the report, investigate it and keep you informed through the
advisory. Please give us a reasonable amount of time to release a fix before you
disclose the problem publicly.

## Supported versions

InferDrift is an early preview. Only the latest commit on the `main` branch
receives fixes.

## Scope

Examples of problems we want to hear about:

- repository code, a pickle or a chat template being executed when the tool says
  it is not;
- an API key appearing in a report, a log, an error message or the output;
- a cost, token or time cap that a run can exceed;
- escape from the sandbox that grades unit-tested code.
