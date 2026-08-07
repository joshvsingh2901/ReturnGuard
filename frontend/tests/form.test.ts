import { describe, expect, it } from "vitest";

import { defaultScenario, toPredictionRequest } from "@/lib/form";

describe("toPredictionRequest", () => {
  it("maps optional product blanks to null without inventing product data", () => {
    const payload = toPredictionRequest(
      {
        ...defaultScenario,
        customerProfileAvailable: true,
        isMale: "false",
        shippingCountry: "Country_A",
        premier: "true",
        productInformationAvailable: true,
        productType: " Jeans ",
        brandDesc: "",
        avgGbpPrice: "",
        avgDiscountValue: "15",
      },
      "private-context-key",
    );

    expect(payload).toEqual({
      customer_profile: { yearOfBirth: null, isMale: false, shippingCountry: "Country_A", premier: true },
      product_profile: { productType: "Jeans", brandDesc: null, avgGbpPrice: null, avgDiscountValue: 15 },
    });
  });

  it("includes an opaque context key only when the customer profile is unavailable", () => {
    const payload = toPredictionRequest(
      { ...defaultScenario, customerProfileAvailable: false, productInformationAvailable: false },
      "private-context-key",
    );

    expect(payload.customer_context_key).toBe("private-context-key");
    expect(payload.customer_profile).toBeNull();
    expect(payload.product_profile).toBeNull();
  });
});
