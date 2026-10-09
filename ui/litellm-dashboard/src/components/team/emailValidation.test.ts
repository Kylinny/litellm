import { describe, expect, it } from "vitest";
import { isEmailish } from "./emailValidation";

describe("isEmailish", () => {
  it("accepts an Entra ID guest UPN with #EXT# in the local part", () => {
    expect(isEmailish("john.doe_contoso.com#EXT#@ourtenant.onmicrosoft.com")).toBe(true);
  });

  it.each([
    "user@example.com",
    "a@b.com",
    "user+tag@example.com",
    "first.last@sub.example.co",
    "user_name@example.com",
  ])("accepts an ordinary address %s", (value) => {
    expect(isEmailish(value)).toBe(true);
  });

  it.each(["!", "$", "%", "&", "*", "/", "=", "?", "^", "`", "{", "|", "}", "~"])(
    "accepts the RFC 5322 atext character %s in the local part",
    (char) => {
      expect(isEmailish(`local${char}part@example.com`)).toBe(true);
    },
  );

  it("accepts an empty string, keeping the field optional", () => {
    expect(isEmailish("")).toBe(true);
  });

  it.each([
    "not-an-email",
    "missing-at.com",
    "@example.com",
    "a@b",
    "a@b.c",
    "a@.com",
    "a@-b.com",
    ".a@example.com",
    "a.@example.com",
    "a..b@example.com",
    "a b@example.com",
    'a"b@example.com',
  ])("rejects the invalid address %s", (value) => {
    expect(isEmailish(value)).toBe(false);
  });
});
