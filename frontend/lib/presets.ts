import type { ScenarioFormValues } from "@/lib/form";

export type DemoPreset = {
  id: "complete" | "customer-missing" | "product-incomplete";
  label: string;
  description: string;
  values: ScenarioFormValues;
};

export const demoPresets: DemoPreset[] = [
  {
    id: "complete",
    label: "Complete profile",
    description: "A synthetic customer and product profile with all supported fields supplied.",
    values: {
      customerProfileAvailable: true,
      yearOfBirth: "1988",
      isMale: "false",
      shippingCountry: "Country_A",
      premier: "true",
      productInformationAvailable: true,
      productType: "Jeans",
      brandDesc: "Brand_A",
      avgGbpPrice: "54.99",
      avgDiscountValue: "15",
    },
  },
  {
    id: "customer-missing",
    label: "Customer profile unavailable",
    description: "A synthetic product profile with the customer profile deliberately unavailable.",
    values: {
      customerProfileAvailable: false,
      yearOfBirth: "",
      isMale: "",
      shippingCountry: "",
      premier: "",
      productInformationAvailable: true,
      productType: "Jeans",
      brandDesc: "Brand_A",
      avgGbpPrice: "54.99",
      avgDiscountValue: "15",
    },
  },
  {
    id: "product-incomplete",
    label: "Incomplete product data",
    description: "A synthetic known customer with optional product attributes left blank.",
    values: {
      customerProfileAvailable: true,
      yearOfBirth: "1994",
      isMale: "true",
      shippingCountry: "Country_B",
      premier: "false",
      productInformationAvailable: true,
      productType: "Jeans",
      brandDesc: "",
      avgGbpPrice: "",
      avgDiscountValue: "15",
    },
  },
];
