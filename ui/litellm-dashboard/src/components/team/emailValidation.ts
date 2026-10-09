import { z } from "zod";

// zod's email pattern rejects '#' and other RFC 5322 atext characters in the local part,
// which Entra ID uses in every guest UPN (e.g. name_contoso.com#EXT#@tenant.onmicrosoft.com)
const ATEXT = "A-Za-z0-9!#$%&'*+/=?^_`{|}~-";
const DOT_ATOM = `[${ATEXT}]+(?:\\.[${ATEXT}]+)*`;
const RFC_5322_EMAIL = new RegExp(
  `^${DOT_ATOM}@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\\.)+[A-Za-z]{2,}$`,
);

export const isEmailish = (value: string): boolean =>
  value === "" || z.email().safeParse(value).success || RFC_5322_EMAIL.test(value);
