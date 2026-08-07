import { z } from "zod";

import type { PredictionRequest } from "@/lib/types";

export type ScenarioFormValues = {
  customerProfileAvailable: boolean;
  yearOfBirth: string;
  isMale: "" | "true" | "false";
  shippingCountry: string;
  premier: "" | "true" | "false";
  productInformationAvailable: boolean;
  productType: string;
  brandDesc: string;
  avgGbpPrice: string;
  avgDiscountValue: string;
};

const optionalNumber = z.string().trim().refine(
  (value) => value === "" || Number.isFinite(Number(value)),
  "Enter a valid number.",
);

export const scenarioSchema = z
  .object({
    customerProfileAvailable: z.boolean(),
    yearOfBirth: optionalNumber,
    isMale: z.enum(["", "true", "false"]),
    shippingCountry: z.string(),
    premier: z.enum(["", "true", "false"]),
    productInformationAvailable: z.boolean(),
    productType: z.string(),
    brandDesc: z.string(),
    avgGbpPrice: optionalNumber,
    avgDiscountValue: optionalNumber,
  })
  .superRefine((values, context) => {
    const birthYear = values.yearOfBirth.trim();
    if (birthYear && (!Number.isInteger(Number(birthYear)) || Number(birthYear) < 1880 || Number(birthYear) > 2026)) {
      context.addIssue({ code: "custom", path: ["yearOfBirth"], message: "Enter a whole year from 1880 to 2026." });
    }
    if (values.customerProfileAvailable) {
      if (!values.shippingCountry.trim()) {
        context.addIssue({ code: "custom", path: ["shippingCountry"], message: "Choose or enter a shipping country." });
      }
      if (!values.isMale) {
        context.addIssue({ code: "custom", path: ["isMale"], message: "Select a value." });
      }
      if (!values.premier) {
        context.addIssue({ code: "custom", path: ["premier"], message: "Select a value." });
      }
    }
    if (values.productInformationAvailable) {
      const price = values.avgGbpPrice.trim();
      const discount = values.avgDiscountValue.trim();
      if (price && Number(price) <= 0) {
        context.addIssue({ code: "custom", path: ["avgGbpPrice"], message: "Average price must be greater than 0." });
      }
      if (discount && (Number(discount) < 0 || Number(discount) > 100)) {
        context.addIssue({ code: "custom", path: ["avgDiscountValue"], message: "Average discount must be from 0 to 100." });
      }
    }
  });

export const defaultScenario: ScenarioFormValues = {
  customerProfileAvailable: true,
  yearOfBirth: "",
  isMale: "",
  shippingCountry: "",
  premier: "",
  productInformationAvailable: true,
  productType: "",
  brandDesc: "",
  avgGbpPrice: "",
  avgDiscountValue: "",
};

const nullIfBlank = (value: string): string | null => value.trim() || null;
const numberIfPresent = (value: string): number | null => (value.trim() ? Number(value) : null);

export function toPredictionRequest(values: ScenarioFormValues, customerContextKey: string): PredictionRequest {
  return {
    customer_context_key: values.customerProfileAvailable ? undefined : customerContextKey,
    customer_profile: values.customerProfileAvailable
      ? {
          yearOfBirth: numberIfPresent(values.yearOfBirth),
          isMale: values.isMale === "true",
          shippingCountry: values.shippingCountry.trim(),
          premier: values.premier === "true",
        }
      : null,
    product_profile: values.productInformationAvailable
      ? {
          productType: nullIfBlank(values.productType),
          brandDesc: nullIfBlank(values.brandDesc),
          avgGbpPrice: numberIfPresent(values.avgGbpPrice),
          avgDiscountValue: numberIfPresent(values.avgDiscountValue),
        }
      : null,
  };
}
